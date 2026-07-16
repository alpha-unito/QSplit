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

import os

import numpy as np

from qsplit.qubo import QUBO


def split_problem(qubo: QUBO) -> list[QUBO]:
    cut_dim = min(int(os.environ["CUT_DIM"]), qubo.problem_size)
    res = []

    mat_abs = np.abs(qubo.mat)
    diag_mask = np.eye(qubo.problem_size, dtype=bool)
    interaction_matrix = mat_abs + mat_abs.T
    interaction_matrix[diag_mask] = 0.0

    for i in range(qubo.problem_size):
        num_neighbors = cut_dim - 1
        best_neighbors = np.argsort(interaction_matrix[i])[-num_neighbors:]
        sub_indices = np.append(best_neighbors, i)
        sub_indices = np.unique(sub_indices)

        sub_mat = qubo.mat[np.ix_(sub_indices, sub_indices)]
        sub_cols = qubo.cols_idx[sub_indices]
        sub_rows = qubo.rows_idx[sub_indices]

        sub_qubo = QUBO(mat=sub_mat, rows_idx=sub_rows, cols_idx=sub_cols)
        res.append(sub_qubo)

    return res
