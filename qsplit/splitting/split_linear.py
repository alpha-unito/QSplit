from qsplit import _core, configuration
from qsplit.qubo import QUBO


@configuration.configured
def split_problem(qubo: QUBO) -> list[QUBO]:
    return _core.split_linear(qubo, int(configuration.require("CUT_DIM")), int(configuration.get("STRIDE", -1)))
