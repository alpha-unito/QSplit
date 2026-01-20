# Copyright (C) 2025  The QSplit Contributors.
# See the 'CONTRIBUTORS' file at the top-level directory of this distribution.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
        return f"QUBO(cols: {self.cols_idx}, rows: {self.rows_idx}, offset: {self.offset}, size: {self.problem_size})"

    @staticmethod
    def sanitize(mat: flt_mat, cols_idx: int_arr, rows_idx: int_arr) -> tuple[flt_mat, int_arr, int_arr]:
        if not np.allclose(mat, np.triu(mat)):
            mat = lu(mat, permute_l=True)[1]

        if mat.shape[0] % 2 == 0:
            return mat, cols_idx, rows_idx

        if cols_idx[-1] == -1 and rows_idx[-1] == -1:
            mat = mat[:-1, :-1]
            cols_idx = cols_idx[:-1]
            rows_idx = rows_idx[:-1]
            return mat, cols_idx, rows_idx

        tmp = np.zeros((mat.shape[0] + 1, mat.shape[1] + 1))
        tmp[:-1, :-1] = mat
        mat = tmp
        cols_idx = np.append(cols_idx, -1)
        rows_idx = np.append(rows_idx, -1)
        return mat, cols_idx, rows_idx
