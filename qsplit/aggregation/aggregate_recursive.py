from qsplit import _core
from qsplit._core import aggregate_trivial as aggregate_solutions_trivial
from qsplit._core import closest as __get_closest_assignments  # noqa: F401
from qsplit._core import combine_rows as __combine_rows  # noqa: F401
from qsplit._core import combine_ul_lr as __combine_ul_lr  # noqa: F401
from qsplit._core import distance as __distance  # noqa: F401
from qsplit._core import fill_inf as __fill_with_inf  # noqa: F401
from qsplit.conflict_resolution.local_search import nan_subqubo
from qsplit.qubo import QUBO

__all__ = ["aggregate_solutions", "aggregate_solutions_trivial", "aggregate_leaves"]


def aggregate_solutions(solutions: tuple[QUBO, QUBO, QUBO], qubo: QUBO) -> QUBO:
    return _core.aggregate_recursive(tuple(solutions), qubo, nan_subqubo)


def aggregate_leaves(solutions: list[QUBO], qubo: QUBO) -> QUBO:
    return _core.aggregate_leaves(solutions, qubo, nan_subqubo)
