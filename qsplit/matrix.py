import numpy as np
from scipy import sparse


def as_csr(matrix):
    matrix = sparse.csr_array(matrix, dtype=np.float64)
    matrix.check_format(full_check=True)
    if not matrix.has_canonical_format:
        matrix = matrix.copy()
        matrix.sum_duplicates()
        matrix.sort_indices()
    return matrix


def sanitize_sparse(matrix):
    matrix = as_csr(matrix).copy()
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1], "Problem matrix must be square"
    lower = sparse.tril(matrix, k=-1, format="csr")
    if np.isnan(matrix.data).any() or np.any(np.abs(lower.data) > 1e-8):
        diagonal = matrix.diagonal()
        matrix = sparse.triu(matrix + matrix.T, k=1, format="csr")
        matrix += sparse.diags_array((diagonal + diagonal) - diagonal, format="csr")
    matrix.data = np.round(matrix.data, 9)
    matrix.data[np.abs(matrix.data) < 1e-12] = 0
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def as_dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def has_terms(matrix):
    return bool(np.any(matrix.data) if sparse.issparse(matrix) else np.any(matrix))


def nonzero_terms(matrix):
    if sparse.issparse(matrix):
        coo = as_csr(matrix).tocoo(copy=False)
        return ((i, j, value) for i, j, value in zip(coo.row, coo.col, coo.data) if value != 0)
    rows, cols = np.nonzero(matrix)
    return zip(rows, cols, matrix[rows, cols])
