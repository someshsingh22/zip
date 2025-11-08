import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import scatter


class WeightedSAGEConv(SAGEConv):
    def forward(self, x, edge_index, edge_weight=None, size=None):
        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])
        self._edge_weight = edge_weight
        out = self.propagate(edge_index, x=x, size=size)
        self._edge_weight = None
        out = self.lin_l(out)
        x_r = x[1]
        out = out + self.lin_r(x_r)
        return out

    def message(self, x_j):
        w = getattr(self, "_edge_weight", None)
        if w is None:
            return x_j
        return x_j * w.view(-1, 1)

class UserSubredditSAGE(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        """GraphSAGE encoder for user-subject interactions.

        Args:
            input_dim: Dimensionality of subreddit input features.
            hidden_dim: Hidden/channel size for projections and SAGE layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sub_proj = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU())
        self.user_proj = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU())
        # Use sum aggregation so that passing per-edge normalized weights yields a weighted mean
        self.conv1 = WeightedSAGEConv((-1, -1), hidden_dim, aggr="sum")
        self.conv2 = WeightedSAGEConv((-1, -1), hidden_dim, aggr="sum")
        self.conv3 = WeightedSAGEConv((-1, -1), hidden_dim, aggr="sum")
        # Maps [norm_log_total_count, comment_frac] -> positive scalar weight
        mlp_hidden = max(32, hidden_dim // 4)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2, mlp_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden, 1),
            torch.nn.Softplus(),  # ensures positive weights
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # Freeze pretrained subreddit embeddings: prevent gradients flowing into the input features
        sub_features = x_dict["subreddit"]
        x_dict["subreddit"] = F.normalize(self.sub_proj(sub_features), dim=-1)

        # Initialize user embeddings if missing (users have no features)
        if "user" not in x_dict:
            # Infer number of users by inspecting edge_index_dict
            max_uid = 0
            for (src_type, rel, dst_type), edge_index in edge_index_dict.items():
                if src_type == "user":
                    max_uid = max(max_uid, int(edge_index[0].max()) + 1)
                elif dst_type == "user":
                    max_uid = max(max_uid, int(edge_index[1].max()) + 1)
            # Initialize with normal distribution
            x_dict["user"] = torch.randn((max_uid, self.input_dim), dtype=x_dict["subreddit"].dtype, device=x_dict["subreddit"].device) * 0.01
        x_dict["user"] = F.normalize(self.user_proj(x_dict["user"]), dim=-1)

        # Message passing over unified 'interacts' edges, reversed to output user embeddings.
        rev_etype = ("subreddit", "rev_interacts", "user")

        edge_index = edge_index_dict[rev_etype]
        edge_weight = None
        # Derive scalar edge weights from edge attributes via MLP and normalize per-destination (user)
        if edge_attr_dict is not None and rev_etype in edge_attr_dict:
            ea = edge_attr_dict[rev_etype]  # [E, 2]
            if ea.dim() == 2 and ea.size(-1) == 2:
                raw_w = self.edge_mlp(ea).view(-1)  # [E], positive
                dst = edge_index[1]
                # Normalize so that sum of weights into each dst node is 1.0
                denom = scatter(raw_w, dst, dim=0, dim_size=x_dict["user"].size(0), reduce="sum")
                edge_weight = raw_w / (denom[dst] + 1e-12)

        u = self.conv1(
            (x_dict["subreddit"], x_dict["user"]),
            edge_index,
            edge_weight=edge_weight,
        ).relu()
        u = self.conv2(
            (x_dict["subreddit"], u),
            edge_index,
            edge_weight=edge_weight,
        ).relu()
        u = self.conv3(
            (x_dict["subreddit"], u),
            edge_index,
            edge_weight=edge_weight,
        )
        x_dict["user"] = F.normalize(u, dim=-1)
        return x_dict


class DotPredictor(torch.nn.Module):
    def __init__(
        self,
        init_logit_scale: float = 0.0,
        max_logit_scale: float = 6.0,
        proj_dim: int | None = None,
        use_mlp: bool = True,
    ):
        super().__init__()
        # Following CLIP, store logit_scale in log space and clamp to a max
        self.logit_scale = torch.nn.Parameter(torch.tensor(init_logit_scale))
        self.max_logit_scale = float(max_logit_scale)
        self.proj_dim = proj_dim
        if proj_dim is None:
            self.user_head = torch.nn.Identity()
            self.sub_head = torch.nn.Identity()
        else:
            if use_mlp:
                self.user_head = torch.nn.Sequential(
                    torch.nn.Linear(proj_dim, proj_dim, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.Linear(proj_dim, proj_dim, bias=True),
                )
                self.sub_head = torch.nn.Sequential(
                    torch.nn.Linear(proj_dim, proj_dim, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.Linear(proj_dim, proj_dim, bias=True),
                )
            else:
                self.user_head = torch.nn.Linear(proj_dim, proj_dim, bias=True)
                self.sub_head = torch.nn.Linear(proj_dim, proj_dim, bias=True)

    def forward(self, src, dst):
        # Pairwise aligned scores (vector). For matrix scores, do src @ dst.t() externally.
        q = F.normalize(self.user_head(src), dim=-1)
        k = F.normalize(self.sub_head(dst), dim=-1)
        scale = self.logit_scale.clamp(max=self.max_logit_scale).exp()
        return scale * (q * k).sum(dim=-1)

    def compute_logits(self, src_batch: torch.Tensor, dst_batch: torch.Tensor, base_tau: float = 1.0) -> torch.Tensor:
        """Compute full [B, B] logits with projections and learnable temperature."""
        q = F.normalize(self.user_head(src_batch), dim=-1)
        k = F.normalize(self.sub_head(dst_batch), dim=-1)
        scale = self.logit_scale.clamp(max=self.max_logit_scale).exp() / float(base_tau)
        return (q @ k.t()) * scale
