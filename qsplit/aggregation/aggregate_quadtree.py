import numpy as np
import pandas as pd

from qsplit.qubo import QUBO


def aggregate_solutions(solutions: list[QUBO], qubo: QUBO) -> QUBO:
    real_ids = sorted(idx for idx in set(qubo.rows_idx).union(qubo.cols_idx) if idx >= 0)
    votes = dict.fromkeys(real_ids, 0.0)
    weights = dict.fromkeys(real_ids, 0.0)

    for sub_qubo in solutions:
        df = sub_qubo.solutions
        finite_solutions = df[np.isfinite(df["energy"])]
        if finite_solutions.empty:
            continue
        best = finite_solutions.nsmallest(1, "energy").iloc[0]
        center = sub_qubo.rows_idx[0]
        for idx in sub_qubo.rows_idx:
            if idx not in votes or idx not in best.index or best[idx] not in (0, 1):
                continue
            weight = 2.0 if idx == center else 1.0
            votes[idx] += weight * best[idx]
            weights[idx] += weight

    assignment = {idx: int(votes[idx] > weights[idx] / 2) for idx in real_ids}
    row_values = np.array([assignment.get(idx, 0) for idx in qubo.rows_idx])
    col_values = np.array([assignment.get(idx, 0) for idx in qubo.cols_idx])
    energy = float(row_values @ qubo.mat @ col_values + qubo.offset)
    qubo.solutions = pd.DataFrame({**{idx: [assignment[idx]] for idx in real_ids}, "energy": [energy]})
    return qubo
