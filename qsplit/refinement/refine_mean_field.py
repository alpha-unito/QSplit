from qsplit import _core, configuration
from qsplit.qubo import QUBO


@configuration.configured
def refine_problems(subproblems: list[QUBO], qubo: QUBO) -> list[QUBO]:
    return _core.refine_mean_field(subproblems, qubo, int(configuration.require("CUT_DIM")))
