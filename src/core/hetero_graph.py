from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData


def load_bipartite_graph_payload(path: str) -> Tuple[Tensor, Optional[Tensor], int, int]:
    """Load the serialized bipartite graph payload produced by prepare_graph.

    Args:
        path: Path to the graph `.pt` file.

    Returns:
        edge_index: LongTensor of shape [2, E] with (user_id, sub_id).
        edge_weight: Optional FloatTensor of shape [E] with edge weights.
        num_users: Number of user nodes.
        num_subs: Number of subreddit nodes.
    """
    payload = torch.load(path, map_location="cpu")
    edge_index = payload["edge_index"].to(torch.long)
    edge_weight = payload.get("edge_weight", None)
    if edge_weight is not None:
        edge_weight = edge_weight.to(torch.float32)
    num_users = int(payload["num_users"])  # rows
    num_subs = int(payload["num_subs"])  # cols
    return edge_index, edge_weight, num_users, num_subs
def load_multi_rel_graph_payload(path: str):
    """Load payload supporting two relations: comments and posts.

    Returns a dict with keys:
      - edge_index_comments, edge_weight_comments
      - edge_index_posts, edge_weight_posts
      - num_users, num_subs
    Falls back to single 'edge_index'/'edge_weight' if multi-rel keys are absent
    by mapping them into the 'comments' relation.
    """
    payload = torch.load(path, map_location="cpu")
    num_users = int(payload["num_users"])  # rows
    num_subs = int(payload["num_subs"])  # cols

    if "edge_index_comments" in payload or "edge_index_posts" in payload:
        eic = payload.get("edge_index_comments", torch.empty(2, 0, dtype=torch.long))
        ewc = payload.get("edge_weight_comments", None)
        if ewc is not None:
            ewc = ewc.to(torch.float32)
        eip = payload.get("edge_index_posts", torch.empty(2, 0, dtype=torch.long))
        ewp = payload.get("edge_weight_posts", None)
        if ewp is not None:
            ewp = ewp.to(torch.float32)
    else:
        # Backward-compatibility: map single relation to comments
        eic = payload["edge_index"].to(torch.long)
        ewc = payload.get("edge_weight", None)
        if ewc is not None:
            ewc = ewc.to(torch.float32)
        eip = torch.empty(2, 0, dtype=torch.long)
        ewp = None

    return {
        "edge_index_comments": eic,
        "edge_weight_comments": ewc,
        "edge_index_posts": eip,
        "edge_weight_posts": ewp,
        "num_users": num_users,
        "num_subs": num_subs,
    }


def build_hetero_from_multi_rel(
    *,
    edge_index_comments: Tensor,
    edge_index_posts: Tensor,
    num_users: int,
    num_subs: int,
    edge_weight_comments: Optional[Tensor] = None,
    edge_weight_posts: Optional[Tensor] = None,
    add_reverse: bool = True,
):
    """Build HeteroData with ('comments') and ('posts') relations and reverses.

    Node types: 'user', 'sub'
    Edge types: ('user','comments','sub'), ('user','posts','sub') and reverse.
    """
    data = HeteroData()
    data["user"].num_nodes = int(num_users)
    data["sub"].num_nodes = int(num_subs)

    # comments
    data[("user", "comments", "sub")].edge_index = edge_index_comments.to(torch.long)
    if edge_weight_comments is not None:
        data[("user", "comments", "sub")].edge_weight = edge_weight_comments.to(torch.float32)
    if add_reverse:
        rev = torch.stack([edge_index_comments[1], edge_index_comments[0]], dim=0)
        data[("sub", "rev_comments", "user")].edge_index = rev
        if edge_weight_comments is not None:
            data[("sub", "rev_comments", "user")].edge_weight = edge_weight_comments.to(torch.float32)

    # posts
    data[("user", "posts", "sub")].edge_index = edge_index_posts.to(torch.long)
    if edge_weight_posts is not None:
        data[("user", "posts", "sub")].edge_weight = edge_weight_posts.to(torch.float32)
    if add_reverse:
        rev = torch.stack([edge_index_posts[1], edge_index_posts[0]], dim=0)
        data[("sub", "rev_posts", "user")].edge_index = rev
        if edge_weight_posts is not None:
            data[("sub", "rev_posts", "user")].edge_weight = edge_weight_posts.to(torch.float32)

    return data


def build_hetero_from_bipartite(
    *,
    edge_index: Tensor,
    num_users: int,
    num_subs: int,
    edge_weight: Optional[Tensor] = None,
    add_reverse: bool = True,
) -> HeteroData:
    """Construct a PyG HeteroData object for a user–subreddit bipartite graph.

    The forward relation is ('user', 'interacts', 'sub'), and when
    `add_reverse=True` a reverse relation ('sub', 'rev_interacts', 'user') is
    added with the same edge indices and (optional) weights.

    Args:
        edge_index: [2, E] (user_id, sub_id) indices.
        num_users: Number of users.
        num_subs: Number of subreddits.
        edge_weight: Optional weights per edge.
        add_reverse: Whether to add the reverse relation.

    Returns:
        A `HeteroData` object ready for neighbor sampling and modeling.
    """
    data = HeteroData()
    data["user"].num_nodes = int(num_users)
    data["sub"].num_nodes = int(num_subs)

    # Forward relation: user -> sub
    data[("user", "interacts", "sub")].edge_index = edge_index.to(torch.long)
    if edge_weight is not None:
        data[("user", "interacts", "sub")].edge_weight = edge_weight.to(torch.float32)

    if add_reverse:
        rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        data[("sub", "rev_interacts", "user")].edge_index = rev_edge_index
        if edge_weight is not None:
            data[("sub", "rev_interacts", "user")].edge_weight = edge_weight.to(torch.float32)

    return data


