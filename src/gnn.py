import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class UserSubredditSAGE(torch.nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, residual: bool = True, heads: int = 5
    ):
        """GATv2 encoder for user-subject interactions.

        Args:
            input_dim: Dimensionality of subreddit input features.
            hidden_dim: Hidden/channel size for projections and SAGE layers.
            residual: Whether to add residual connections between graph layers.
            heads: Number of heads for the GATv2Conv layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.residual = bool(residual)
        self.sub_proj = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU()
        )
        self.user_proj = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU()
        )
        # Use GATv2 with multi-head attention; keep output size == hidden_dim
        # Set add_self_loops=False for bipartite edges
        self.conv1 = GATv2Conv(
            (-1, -1),
            hidden_dim,
            heads=heads,
            concat=False,
            add_self_loops=False,
            edge_dim=2,
        )
        self.conv2 = GATv2Conv(
            (-1, -1),
            hidden_dim,
            heads=heads,
            concat=False,
            add_self_loops=False,
            edge_dim=2,
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
            x_dict["user"] = (
                torch.randn(
                    (max_uid, self.input_dim),
                    dtype=x_dict["subreddit"].dtype,
                    device=x_dict["subreddit"].device,
                )
                * 0.01
            )
        x_dict["user"] = F.normalize(self.user_proj(x_dict["user"]), dim=-1)

        # Message passing over unified 'interacts' edges, reversed to output user embeddings.
        rev_etype = ("subreddit", "rev_interacts", "user")

        edge_index = edge_index_dict[rev_etype]
        edge_attr = None
        # Provide raw edge attributes directly to GATv2 (edge_dim=2)
        if edge_attr_dict is not None and rev_etype in edge_attr_dict:
            ea = edge_attr_dict[rev_etype]  # [E, 2]
            if ea is not None and ea.dim() == 2:
                edge_attr = ea
        # Layer 1
        u1 = self.conv1(
            (x_dict["subreddit"], x_dict["user"]),
            edge_index,
            edge_attr=edge_attr,
        )
        if self.residual:
            u1 = u1 + x_dict["user"]
        u1 = u1.relu()
        # Layer 2
        u2 = self.conv2(
            (x_dict["subreddit"], u1),
            edge_index,
            edge_attr=edge_attr,
        )
        if self.residual:
            u2 = u2 + u1
        x_dict["user"] = F.normalize(u2, dim=-1)
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

    def forward(self, src, dst):
        # Pairwise aligned scores (vector). For matrix scores, do src @ dst.t() externally.
        q = F.normalize(self.user_head(src), dim=-1)
        k = F.normalize(self.sub_head(dst), dim=-1)
        scale = self.logit_scale.clamp(max=self.max_logit_scale).exp()
        return scale * (q * k).sum(dim=-1)

    def compute_logits(
        self, src_batch: torch.Tensor, dst_batch: torch.Tensor, base_tau: float = 1.0
    ) -> torch.Tensor:
        """Compute full [B, B] logits with projections and learnable temperature."""
        q = F.normalize(self.user_head(src_batch), dim=-1)
        k = F.normalize(self.sub_head(dst_batch), dim=-1)
        scale = self.logit_scale.clamp(max=self.max_logit_scale).exp() / float(base_tau)
        return (q @ k.t()) * scale


class TextProjector(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        """Project external text embeddings into the GNN hidden space.

        Args:
            input_dim: Dimensionality of the input text embeddings.
            output_dim: Target hidden size to align with user/subreddit embeddings.
        """
        super().__init__()
        self.proj = torch.nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize text embeddings.

        Args:
            text_embeddings: Tensor of shape [N, input_dim].

        Returns:
            Tensor of shape [N, output_dim], L2-normalized.
        """
        x = self.proj(text_embeddings)
        return F.normalize(x, dim=-1)
