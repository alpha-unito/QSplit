import numpy as np
import pandas as pd
from scipy.linalg import lu

int_arr = np.ndarray[tuple[int], np.dtype[int]]
flt_mat = np.ndarray[tuple[int, int], np.dtype[np.float64]]


class QUBO:
    def __init__(self, mat: flt_mat, rows_idx: int_arr, cols_idx: int_arr, offset: float = 0.0):
        assert mat.shape[0] == mat.shape[1], "Problem matrix must be square"
        assert mat.shape[1] == cols_idx.shape[0], "Invalid numer of columns, must be as long as cols_idx array"
        assert mat.shape[0] == rows_idx.shape[0], "Invalid numer of rows, must be as long as rows_idx array"
        mat, cols_idx, rows_idx = QUBO.sanitize(mat, cols_idx, rows_idx)
        self.mat: np.ndarray[tuple[int, int], np.dtype[np.float64]] = mat
        self.cols_idx: np.ndarray[tuple[int], np.dtype[int]] = cols_idx
        self.rows_idx: np.ndarray[tuple[int], np.dtype[int]] = rows_idx
        self.offset: float = offset
        self.problem_size: int = mat.shape[0]
        self.solutions: pd.DataFrame | None = None

    def __str__(self) -> str:
        return f"QUBO(cols: {self.cols_idx}, rows: {self.rows_idx}, offset: {self.offset}, problem_size: {self.problem_size})"

    @staticmethod
    def sanitize(mat: flt_mat, cols_idx: int_arr, rows_idx: int_arr) -> tuple[flt_mat, int_arr, int_arr]:
        if not np.allclose(mat, np.triu(mat)):
            mat = lu(mat, permute_l=True)[1]

        if mat.shape[0] % 2 != 0:
            '''
            TODO check last index of rows and cols, if both are -1 instead of adding another -1 contract the matrix
            
            1  2  -1    1 2                 1  2  -1 -1
            3  4  -1 -> 3 4 - INSTEAD OF -> 3  4  -1 -1
            -1 -1 -1                        -1 -1 -1 -1
                                            -1 -1 -1 -1
            '''
            tmp = np.zeros((mat.shape[0] + 1, mat.shape[1] + 1))
            tmp[:-1, :-1] = mat
            mat = tmp
            cols_idx = np.append(cols_idx, -1)
            rows_idx = np.append(rows_idx, -1)

        return mat, cols_idx, rows_idx

    # def __eq__(self, other)  # def __repr__(self)
