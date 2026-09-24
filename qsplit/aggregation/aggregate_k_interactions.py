import numpy as np
import pandas as pd

from qsplit.qubo import QUBO


def aggregate_solutions(solutions: list[QUBO], qubo: QUBO) -> QUBO:
    votes = {idx: [] for idx in qubo.rows_idx if idx >= 0}
    weights = {idx: [] for idx in qubo.rows_idx if idx >= 0}

    for center_idx, sub_qubo in enumerate(solutions):
        df_sol = sub_qubo.solutions
        best_sol = df_sol.nsmallest(1, "energy").iloc[0]
        for col in df_sol.columns:
            if col == "energy" or col < 0:
                continue
            val = best_sol[col]
            votes[col].append(val)

            weight = 2.0 if col == center_idx else 1.0
            weights[col].append(weight)

    assignment = {}
    for idx in votes:
        if len(votes[idx]) > 0:
            weighted_avg = np.average(votes[idx], weights=weights[idx])
            assignment[idx] = int(round(weighted_avg))
        else:
            assignment[idx] = 0

    rows = np.array([assignment.get(idx, 0) for idx in qubo.rows_idx])
    cols = np.array([assignment.get(idx, 0) for idx in qubo.cols_idx])
    energy = rows @ qubo.mat @ cols + qubo.offset
    sol_dict = {idx: [assignment[idx]] for idx in sorted(assignment.keys())}
    sol_dict["energy"] = [float(energy)]
    qubo.solutions = pd.DataFrame(sol_dict)
    return qubo
