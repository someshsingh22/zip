import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import torch.nn as nn

class GATBlock(nn.Module):
    '''GATv2 MHA block with LayerNorm. No self loops for bipartite edges
    
    Args:
        hidden_dim: Hidden dimension of the GATv2 block
        heads: Number of heads for the GATv2Conv layer
        edge_dim: Dimension of the edge attributes
        dropout: Dropout rate
    '''
    def __init__(self, hidden_dim, heads, edge_dim=2, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.conv = GATv2Conv(
            (-1, -1),
            hidden_dim,
            heads=heads,
            concat=False,
            add_self_loops=False,
            edge_dim=edge_dim,
            negative_slope=0.2,
            dropout=dropout,
            bias=True
        )
    
    def forward(self, x, edge_index, edge_attr):
        return self.conv((self.norm(x[0]), self.norm(x[1])), edge_index, edge_attr)
    
class NodeProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1, activation=nn.LeakyReLU(0.2)):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation,
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.proj(x)

class RedditGATv2(torch.nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, residual: bool = True, heads: int = 5
    ):
        """GATv2 encoder for Reddit interactions.

        Args:
            input_dim: Dimensionality of subreddit input features (pretrained subreddit description embeddings).
            hidden_dim: Hidden/channel size for projections and GATv2 layers.
            residual: Whether to add residual (skip) connections between graph layers.
            heads: Number of heads for the GATv2Conv layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.residual = bool(residual)
        self.heads = heads
        
        self.sub_proj = NodeProjection(input_dim, hidden_dim)
        self.user_proj = NodeProjection(input_dim, hidden_dim)
        self.conv1 = GATBlock(hidden_dim, heads)
        self.conv2 = GATBlock(hidden_dim, heads)
        
    def init_user_embeddings(self, edge_index_dict, d_type, device):
        max_uid = 0
        for (src_type, rel, dst_type), edge_index in edge_index_dict.items():
            if src_type == "user":
                max_uid = max(max_uid, int(edge_index[0].max()) + 1)
            elif dst_type == "user":
                max_uid = max(max_uid, int(edge_index[1].max()) + 1)
        return torch.randn((max_uid, self.input_dim), dtype=d_type, device=device) * 0.01

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        sub_features = x_dict["subreddit"]
        x_dict["subreddit"] = F.normalize(self.sub_proj(sub_features), dim=-1)

        if "user" not in x_dict:
            x_dict["user"] = self.init_user_embeddings(edge_index_dict, sub_features.dtype, sub_features.device)
        x_dict["user"] = F.normalize(self.user_proj(x_dict["user"]), dim=-1)

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
            torch.nn.SiLU(),
            torch.nn.Linear(proj_dim, proj_dim, bias=True),
        )
        self.sub_head = torch.nn.Sequential(
            torch.nn.Linear(proj_dim, proj_dim, bias=True),
            torch.nn.SiLU(),
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


class TextProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),         # normalize variance across dims
            nn.GELU(),                        # smooth nonlinearity, preserves sign
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),         # stabilize before normalization
        )

    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        x = self.proj(text_embeddings)
        return F.normalize(x, dim=-1)
