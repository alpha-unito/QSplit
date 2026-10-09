from qsplit import configuration
from qsplit._core import is_empty as is_empty
from qsplit._core import vars_count as vars_count
from qsplit.qubo import QUBO


@configuration.configured
def is_sparse(qubo: QUBO, cut_dim=None) -> bool:
    return vars_count(qubo) <= int(cut_dim or configuration.require("CUT_DIM"))
