import numpy as np

from qsplit.qubo import QUBO


def condition_subproblem(qubo: QUBO, window: list[int], beliefs: dict[int, float]) -> QUBO:
    outside = [idx for idx in qubo.rows_idx if idx >= 0 and idx not in window]
    row_map = {idx: pos for pos, idx in enumerate(qubo.rows_idx)}
    col_map = {idx: pos for pos, idx in enumerate(qubo.cols_idx)}
    rows = [row_map[idx] for idx in window]
    cols = [col_map[idx] for idx in window]
    outside_rows = [row_map[idx] for idx in outside]
    outside_cols = [col_map[idx] for idx in outside]
    probabilities = np.array([beliefs[idx] for idx in outside])
    mat = qubo.mat[np.ix_(rows, cols)].astype(float, copy=True)
    coupling = qubo.mat[np.ix_(rows, outside_cols)] + qubo.mat[np.ix_(outside_rows, cols)].T
    mat[np.diag_indices_from(mat)] += coupling @ probabilities

    outside_mat = qubo.mat[np.ix_(outside_rows, outside_cols)]
    offset = qubo.offset + probabilities @ outside_mat @ probabilities
    offset += np.diag(outside_mat) @ (probabilities - probabilities**2)
    ids = np.array(window, dtype=int)
    return QUBO(mat, ids.copy(), ids.copy(), offset=float(offset))


def collect_beliefs(subproblems: list[QUBO], qubo: QUBO) -> dict[int, float]:
    best = qubo.solutions.nsmallest(1, "energy").iloc[0]
    beliefs = {idx: float(best[idx]) for idx in set(qubo.rows_idx) | set(qubo.cols_idx) if idx >= 0}
    votes = {idx: [] for idx in beliefs}
    for sub in subproblems:
        df = sub.solutions
        if df is None or df.empty or "energy" not in df:
            continue
        finite = df[np.isfinite(df["energy"])]
        if finite.empty:
            continue
        local_best = finite[finite["energy"] == finite["energy"].min()]
        for idx in (set(sub.rows_idx) | set(sub.cols_idx)) & beliefs.keys():
            if idx not in local_best:
                continue
            values = local_best.loc[local_best[idx].isin([0, 1]), idx]
            if not values.empty:
                votes[idx].append(float(values.mean()))
    for idx, values in votes.items():
        if values:
            beliefs[idx] = (beliefs[idx] + float(np.mean(values))) / 2.0
    return beliefs
