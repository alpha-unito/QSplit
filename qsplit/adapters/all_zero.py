import pandas as pd

from qsplit.qubo import QUBO


def solve(qubo: QUBO) -> pd.DataFrame:
    all_indices = sorted(list(set(qubo.rows_idx).union(qubo.cols_idx)))
    data = [[0 for _ in range(len(all_indices) + 1)]]
    return pd.DataFrame(data, columns=all_indices + ['energy'])