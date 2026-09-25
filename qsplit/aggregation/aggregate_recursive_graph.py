from collections import Counter

import numpy as np
import pandas as pd

from qsplit.qubo import QUBO


def aggregate_solutions(solutions: list[QUBO], qubo: QUBO) -> QUBO:
    all_indices = sorted([idx for idx in set(qubo.rows_idx) if idx >= 0])
    votes = {idx: [] for idx in all_indices}

    for sub_qubo in solutions:
        df_sol = sub_qubo.solutions
        best_sol = df_sol.nsmallest(1, "energy").iloc[0]
        for col in df_sol.columns:
            if col == "energy" or col < 0:
                continue
            elif col in votes and best_sol[col] in (0, 1):
                votes[col].append(best_sol[col])

    assignment = {}
    for idx in all_indices:
        var_votes = votes[idx]
        if len(var_votes) > 0:
            most_common_val, _ = Counter(var_votes).most_common(1)[0]
            assignment[idx] = int(round(most_common_val))
        else:
            assignment[idx] = 0

    rows = np.array([assignment.get(idx, 0) for idx in qubo.rows_idx])
    cols = np.array([assignment.get(idx, 0) for idx in qubo.cols_idx])
    energy = rows @ qubo.mat @ cols + qubo.offset
    sol_dict = {idx: [assignment[idx]] for idx in all_indices}
    sol_dict["energy"] = [float(energy)]
    qubo.solutions = pd.DataFrame(sol_dict)
    return qubo
