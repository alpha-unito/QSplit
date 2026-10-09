from qsplit import _core
from qsplit.qubo import QUBO


def aggregate_solutions(solutions: list[QUBO], qubo: QUBO) -> QUBO:
    return _core.aggregate_votes(solutions, qubo, "quadtree")
