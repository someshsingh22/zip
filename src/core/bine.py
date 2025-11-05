from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class BipartiteEmbeddingModel(nn.Module):
    """Simple BiNE-style bipartite embedding model with negative sampling.

    This is a lightweight fallback when a dedicated BiNE implementation is not available.
    It learns two embedding tables (users, subreddits) and optimizes a logistic
    loss over positive and negative pairs.
    """

    def __init__(
        self,
        num_users: int,
        num_subs: int,
        embedding_dim: int,
        init_sub_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.sub_emb = nn.Embedding(num_subs, embedding_dim)

        # Initialization
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.sub_emb.weight, mean=0.0, std=0.02)
        if init_sub_weight is not None:
            assert (
                init_sub_weight.shape == self.sub_emb.weight.data.shape
            ), "init_sub_weight shape must match subreddit embedding table"
            with torch.no_grad():
                self.sub_emb.weight.copy_(init_sub_weight)

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_sub_ids: torch.Tensor,
        neg_sub_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute logits for positive and negative pairs.

        Args:
            user_ids: [B]
            pos_sub_ids: [B]
            neg_sub_ids: [B, K]

        Returns:
            pos_logits: [B]
            neg_logits: [B, K]
        """
        u = self.user_emb(user_ids)  # [B, D]
        sp = self.sub_emb(pos_sub_ids)  # [B, D]
        sn = self.sub_emb(neg_sub_ids)  # [B, K, D]

        pos_logits = (u * sp).sum(dim=1)
        # [B, K]
        neg_logits = torch.einsum("bd,bkd->bk", u, sn)
        return pos_logits, neg_logits

    @staticmethod
    def loss_fn(
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        pos_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Logistic loss with edge weighting and K negatives.

        L = w * [ softplus(-pos) + sum_k softplus(neg_k) ]
        """
        pos_term = F.softplus(-pos_logits)
        neg_term = F.softplus(neg_logits).sum(dim=1)
        loss = (pos_term + neg_term) * pos_weight
        return loss.mean()


