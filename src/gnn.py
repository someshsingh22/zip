import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class UserSubredditSAGE(torch.nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=512):
        super().__init__()
        self.sub_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.conv1 = SAGEConv((-1, -1), hidden_dim)
        self.conv2 = SAGEConv((-1, -1), hidden_dim)
        # Maps [log_total_count, comment_frac, post_frac] -> positive scalar weight
        mlp_hidden = max(32, hidden_dim // 4)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(3, mlp_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden, 1),
            torch.nn.Softplus(),  # ensures positive weights
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # Ensure subreddit embeddings exist
        if "subreddit" not in x_dict:
            raise KeyError("Missing subreddit nodes in batch")
        x_dict["subreddit"] = F.normalize(self.sub_proj(x_dict["subreddit"]), dim=-1)

        # Initialize user embeddings if missing (users have no features)
        if "user" not in x_dict:
            # Infer number of users by inspecting edge_index_dict
            max_uid = 0
            for (src_type, rel, dst_type), edge_index in edge_index_dict.items():
                if src_type == "user":
                    max_uid = max(max_uid, int(edge_index[0].max()) + 1)
                elif dst_type == "user":
                    max_uid = max(max_uid, int(edge_index[1].max()) + 1)
            if max_uid == 0:
                raise KeyError("No user nodes found in sampled subgraph")
            x_dict["user"] = torch.zeros((max_uid, self.sub_proj.out_features),
                                         dtype=x_dict["subreddit"].dtype,
                                         device=x_dict["subreddit"].device)

        # Message passing over unified 'interacts' edges, reversed to output user embeddings.
        rev_etype = ("subreddit", "rev_interacts", "user")
        if rev_etype not in edge_index_dict:
            raise RuntimeError("No 'rev_interacts' edges found in this mini-batch")

        edge_index = edge_index_dict[rev_etype]
        # Derive scalar edge weights from edge attributes (fallback to ones)
        edge_attr = None
        if hasattr(self, "edge_mlp"):
            # Edge attributes should be present in the HeteroData for rev_interacts
            # If missing, default to ones
            edge_attr = x_dict.get("__edge_attr_cache__", None)
        # Extract edge_attr from the dedicated store if available
        # Access pattern: It is stored on the batch object, not in x_dict, so we cannot fetch here.
        # Instead, rely on the LinkNeighborLoader keeping edge_attr in the store; PyG SAGEConv expects
        # edge_weight as a tensor aligned with edge_index.
        # We fetch from edge_index_dict via a side channel is not available, so we rely on a convention:
        # The caller passes a batch where rev_interacts store has 'edge_attr'.
        # To keep this module self-contained, we will not depend on x_dict; instead, we look up a parallel
        # structure passed in during forward via a special key on edge_index_dict if provided.
        # If not provided, we compute uniform weights.
        edge_weight = None
        if edge_attr_dict is not None and rev_etype in edge_attr_dict:
            ea = edge_attr_dict[rev_etype]
            if ea.dim() == 1:
                edge_weight = ea
            else:
                edge_weight = self.edge_mlp(ea).view(-1)
        else:
            edge_weight = None  # uniform weights

        u = self.conv1(
            (x_dict["subreddit"], x_dict["user"]),
            edge_index,
            edge_weight=edge_weight,
        ).relu()
        u = self.conv2(
            (x_dict["subreddit"], u),
            edge_index,
            edge_weight=edge_weight,
        )

        x_dict["user"] = F.normalize(u, dim=-1)
        return x_dict


class DotPredictor(torch.nn.Module):
    def forward(self, src, dst):
        return (src * dst).sum(dim=-1)
