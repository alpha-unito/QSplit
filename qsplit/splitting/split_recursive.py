import numpy as np

from qsplit.qubo import QUBO


def split_problem(qubo: QUBO) -> tuple[QUBO, QUBO, QUBO]:
    """
    Splits the QUBO matrix into three sub-problems:
    - Upper left
    - Upper right
    - Lower right

    If the matrix has an odd dimension, adds one row and
    one column of zero padding before splitting.
    The original QUBO is not modified.
    """
    n = qubo.problem_size
    mat = qubo.mat
    rows_idx = qubo.rows_idx
    cols_idx = qubo.cols_idx

    if n % 2 != 0:
        mat = np.pad(mat, ((0, 1), (0, 1)), mode="constant", constant_values=0)
        rows_idx = np.append(rows_idx, -1)
        cols_idx = np.append(cols_idx, -1)
        n += 1

    split_idx = n // 2
    if not np.all(mat[split_idx:, :split_idx] == 0):
        raise ValueError("Lower left sub-matrix must be 0")

    ul = QUBO(mat[:split_idx, :split_idx], cols_idx=cols_idx[:split_idx], rows_idx=rows_idx[:split_idx])
    ur = QUBO(mat[:split_idx, split_idx:], cols_idx=cols_idx[split_idx:], rows_idx=rows_idx[:split_idx])
    lr = QUBO(mat[split_idx:, split_idx:], cols_idx=cols_idx[split_idx:], rows_idx=rows_idx[split_idx:])
    return ul, ur, lr


def split_leaves(qubo: QUBO) -> list[QUBO]:
    import os
    from copy import deepcopy

    from qsplit.halting_heuristic.stop import is_empty, is_sparse

    cut_dim = int(os.environ["CUT_DIM"])
    leaves = []

    def visit(node):
        if node.problem_size <= cut_dim or is_empty(node) or is_sparse(node):
            leaves.append(node)
            return len(leaves) - 1
        return node, tuple(visit(sub) for sub in split_problem(node))

    qubo.split_tree = visit(deepcopy(qubo))
    return leaves
