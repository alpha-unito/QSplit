from collections.abc import Callable

import numpy as np
import pandas as pd

from qsplit.qubo import QUBO
from qsplit.refinement.propagation import condition_subproblem


def refine_solutions(
    qubo: QUBO,
    solve: Callable[[QUBO], pd.DataFrame],
    block_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if block_size <= 0:
        raise ValueError("Refinement block_size must be positive")
    current = qubo.solutions.nsmallest(1, "energy").iloc[0].copy()
    variables = [idx for idx in qubo.rows_idx if idx >= 0]
    order = rng.permutation(variables)
    for start in range(0, len(order), block_size):
        window = order[start : start + block_size].tolist()
        sub = condition_subproblem(qubo, window, current.to_dict())
        if not np.any(sub.mat):
            continue
        samples = solve(sub)
        for _, sample in samples.iterrows():
            values = sample.reindex(window).to_numpy()
            if not np.isin(values, [0, 1]).all():
                raise ValueError("Conditioned refinement requires binary samples for every local variable")
            candidate = current.copy()
            candidate.loc[window] = values
            rows = np.array([candidate[idx] if idx >= 0 else 0 for idx in qubo.rows_idx])
            cols = np.array([candidate[idx] if idx >= 0 else 0 for idx in qubo.cols_idx])
            energy = float(rows @ qubo.mat @ cols + qubo.offset)
            if energy <= current["energy"]:
                candidate["energy"] = energy
                current = candidate
    return pd.DataFrame([current]).reset_index(drop=True)
