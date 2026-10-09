from qsplit.aggregation.aggregate_k_interactions import aggregate_solutions as aggregate_interactions
from qsplit.aggregation.aggregate_linear import aggregate_solutions as aggregate_linear
from qsplit.aggregation.aggregate_linear_belief_propagation import aggregate_solutions as aggregate_bp
from qsplit.aggregation.aggregate_quadtree import aggregate_solutions as aggregate_quadtree
from qsplit.aggregation.aggregate_recursive_graph import aggregate_solutions as aggregate_graph
from qsplit.splitting.split_k_interactions import split_problem as split_interactions
from qsplit.splitting.split_linear import split_problem as split_linear
from qsplit.splitting.split_quadtree import split_problem as split_quadtree
from qsplit.splitting.split_recursive_graph import split_problem as split_graph

SPLITTERS = {
    "linear": split_linear,
    "k_interactions": split_interactions,
    "quadtree": split_quadtree,
    "recursive_graph": split_graph,
}
AGGREGATORS = {
    "linear": aggregate_linear,
    "linear_belief_propagation": aggregate_bp,
    "recursive_graph": aggregate_graph,
    "k_interactions": aggregate_interactions,
    "quadtree": aggregate_quadtree,
}
SPLIT_METHODS = ["recursive", *SPLITTERS]
AGGREGATE_METHODS = ["auto", "recursive", *AGGREGATORS]


def resolve_aggregate(split_method: str, aggregate_method: str) -> str:
    if split_method not in SPLIT_METHODS:
        raise ValueError(f"Unknown split method: {split_method}")
    method = split_method if aggregate_method == "auto" else aggregate_method
    if method not in AGGREGATE_METHODS:
        raise ValueError(f"Unknown aggregate method: {method}")
    if method in {"recursive", "k_interactions", "quadtree"} and method != split_method:
        raise ValueError(f"{method} aggregation is incompatible with {split_method} splitting")
    return method


def resolve_refinement_aggregate(method: str) -> str:
    if method not in {"linear", "linear_belief_propagation", "recursive_graph"}:
        raise ValueError("Refinement aggregation must be linear, linear_belief_propagation or recursive_graph")
    return method
