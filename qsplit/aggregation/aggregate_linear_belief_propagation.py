import numpy as np
import pandas as pd

from qsplit.qubo import QUBO


def aggregate_solutions(solutions: list[QUBO], qubo: QUBO) -> QUBO:
    all_indices = sorted([idx for idx in set(qubo.rows_idx).union(qubo.cols_idx) if idx >= 0])
    idx_to_pos = {idx: i for i, idx in enumerate(all_indices)}
    beliefs_1 = np.zeros(len(all_indices))
    beliefs_0 = np.zeros(len(all_indices))

    for sub_qubo in solutions:
        df = sub_qubo.solutions
        valid_columns = [col for col in df.columns if col in idx_to_pos]
        min_energy = df["energy"].min()
        energies = df["energy"].to_numpy()
        weights = np.exp(-(energies - min_energy))
        weights /= np.sum(weights)

        for idx_row, (_, row) in enumerate(df.iterrows()):
            w = weights[idx_row]
            for col in valid_columns:
                pos = idx_to_pos[col]
                val = row[col]
                beliefs_1[pos] += val * w
                beliefs_0[pos] += (1.0 - val) * w

    x = np.array([int(beliefs_1[i] >= beliefs_0[i]) for i in range(len(all_indices))])
    assignment = dict(zip(all_indices, x))
    rows = np.array([assignment.get(idx, 0) for idx in qubo.rows_idx])
    cols = np.array([assignment.get(idx, 0) for idx in qubo.cols_idx])
    energy = rows @ qubo.mat @ cols + qubo.offset
    sol_dict = {all_indices[i]: [x[i]] for i in range(len(all_indices))}
    sol_dict["energy"] = [float(energy)]
    qubo.solutions = pd.DataFrame(sol_dict)
    return qubo
