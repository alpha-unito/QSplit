import os

import numpy as np

from qsplit.qubo import QUBO
from qsplit.refinement.propagation import collect_beliefs


def refine_problems(subproblems: list[QUBO], qubo: QUBO) -> list[QUBO]:
    strength = float(os.environ.get("REFINEMENT_STRENGTH", "0.1"))
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("REFINEMENT_STRENGTH must be finite and non-negative")
    beliefs = collect_beliefs(subproblems, qubo)
    row_map = {idx: pos for pos, idx in enumerate(qubo.rows_idx)}
    col_map = {idx: pos for pos, idx in enumerate(qubo.cols_idx)}
    refined = []
    for sub in subproblems:
        macro_members = getattr(sub, "macro_members", {})
        row_projection = np.zeros((len(qubo.rows_idx), sub.problem_size))
        col_projection = np.zeros((len(qubo.cols_idx), sub.problem_size))
        targets = []
        for pos, idx in enumerate(sub.rows_idx):
            members = [idx] if idx >= 0 else macro_members[idx]
            targets.append(float(np.mean([beliefs[member] for member in members])))
            for member in members:
                row_projection[row_map[member], pos] = 1.0
                col_projection[col_map[member], pos] = 1.0
        base = QUBO(
            row_projection.T @ qubo.mat @ col_projection,
            sub.rows_idx.copy(),
            sub.cols_idx.copy(),
            offset=qubo.offset,
        )
        targets = np.array(targets)
        magnitudes = np.abs(base.mat)
        scale = magnitudes.sum(axis=0) + magnitudes.sum(axis=1) - np.diag(magnitudes)
        penalties = strength * scale
        base.mat[np.diag_indices_from(base.mat)] += penalties * (1.0 - 2.0 * targets)
        base.offset += float(penalties @ (targets**2))
        base.macro_members = {idx: list(members) for idx, members in macro_members.items()}
        refined.append(base)
    return refined
