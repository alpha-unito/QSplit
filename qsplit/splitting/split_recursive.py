from qsplit import _core, configuration
from qsplit.qubo import QUBO


@configuration.configured
def split_problem(qubo: QUBO) -> tuple[QUBO, QUBO, QUBO]:
    return _core.split_recursive(qubo)


@configuration.configured
def split_leaves(qubo: QUBO) -> list[QUBO]:
    return _core.split_leaves(qubo, int(configuration.require("CUT_DIM")))
