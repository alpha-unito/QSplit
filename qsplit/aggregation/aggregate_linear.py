import numpy as np
import pandas as pd

from qsplit.qubo import QUBO


def aggregate_solutions(solutions: list[QUBO], qubo: QUBO) -> QUBO:
    all_indices = sorted([idx for idx in set(qubo.rows_idx).union(qubo.cols_idx) if idx >= 0])
    idx_to_pos = {idx: i for i, idx in enumerate(all_indices)}
    out_solutions = [0.0] * len(all_indices)
    counts = [0] * len(all_indices)

    for sub_qubo in solutions:
        df = sub_qubo.solutions
        valid_columns = [col for col in df.columns if col in idx_to_pos]

        for _, row in df.iterrows():
            for col in valid_columns:
                pos = idx_to_pos[col]
                out_solutions[pos] += row[col]
                counts[pos] += 1

    for i in range(len(out_solutions)):
        if counts[i] > 0:
            out_solutions[i] = out_solutions[i] / counts[i]

    x = np.array([round(val) for val in out_solutions])
    n = len(all_indices)
    energy = x.T @ qubo.mat[:n, :n] @ x
    sol_dict = {all_indices[i]: [x[i]] for i in range(len(all_indices))}
    sol_dict["energy"] = [float(energy)]

    qubo.solutions = pd.DataFrame(sol_dict)

    return qubo
