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
    cut_dim = int(os.environ["CUT_DIM"])
    padding_needed = cut_dim - (qubo.problem_size % cut_dim)

    if padding_needed > 0:
        qubo.mat = np.pad(qubo.mat, ((0, padding_needed), (0, padding_needed)), mode="constant", constant_values=0)
        new_indices = np.arange(-1, -(padding_needed + 1), -1)
        qubo.rows_idx = np.concatenate([qubo.rows_idx, new_indices])
        qubo.cols_idx = np.concatenate([qubo.cols_idx, new_indices])
        qubo.problem_size = qubo.mat.shape[0]

    res = []
    for i in range(0, qubo.problem_size, cut_dim):
        end = i + cut_dim
        mat = qubo.mat[i:end, i:end]
        if np.count_nonzero(mat) == 0:
            continue
        sub_qubo = QUBO(mat, qubo.rows_idx[i:end], qubo.cols_idx[i:end])
        res.append(sub_qubo)

    return res
