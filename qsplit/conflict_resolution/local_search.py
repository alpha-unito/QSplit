import pandas as pd

from qsplit import _core
from qsplit._core import calculate_energy as calculate_energy
from qsplit.qubo import QUBO

EXACT_CONFLICT_LIMIT = 10


def solve(qubo: QUBO) -> pd.DataFrame:
    # Only large conflicts need the optional Python sampler.
    from qsplit.adapters.dwave.dwave_sa import solve as sample

    return sample(qubo)


def nan_subqubo(df: pd.DataFrame, qubo: QUBO) -> pd.DataFrame:
    return _core.resolve_conflicts(df, qubo, solve, EXACT_CONFLICT_LIMIT)
