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
        # Optional: derive scalar edge weights from edge attributes via MLP.
        # Current SAGEConv version here does not accept edge_weight; we compute but do not pass it.
        if edge_attr_dict is not None and rev_etype in edge_attr_dict:
            ea = edge_attr_dict[rev_etype]  # [E, 3]
            if ea.dim() == 2 and ea.size(-1) == 3:
                _ = self.edge_mlp(ea).view(-1)  # reserved for future use

        u = self.conv1(
            (x_dict["subreddit"], x_dict["user"]),
            edge_index,
        ).relu()
        u = self.conv2(
            (x_dict["subreddit"], u),
            edge_index,
        )

        x_dict["user"] = F.normalize(u, dim=-1)
        return x_dict


class DotPredictor(torch.nn.Module):
    def forward(self, src, dst):
        return (src * dst).sum(dim=-1)
