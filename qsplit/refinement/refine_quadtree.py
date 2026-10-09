from qsplit import _core, configuration
from qsplit.qubo import QUBO


@configuration.configured
def refine_problems(subproblems: list[QUBO], qubo: QUBO) -> list[QUBO]:
    return _core.refine_quadtree(subproblems, qubo, float(configuration.get("REFINEMENT_STRENGTH", "0.1")))
