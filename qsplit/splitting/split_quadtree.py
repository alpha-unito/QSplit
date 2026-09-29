from qsplit import _core, configuration
from qsplit.qubo import QUBO


@configuration.configured
def split_problem(qubo: QUBO) -> list[QUBO]:
    return _core.split_quadtree(
        qubo, int(configuration.require("CUT_DIM")), float(configuration.get("EXACT_RATIO", "0.75"))
    )
