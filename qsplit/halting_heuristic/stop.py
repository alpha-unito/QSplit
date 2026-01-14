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

def vars_count(qubo: QUBO) -> int:
    rows_found, cols_found = np.nonzero(qubo.mat)
    variables_in_rows = qubo.rows_idx[rows_found]
    variables_in_cols = qubo.cols_idx[cols_found]
    unique_vars = np.unique(np.concatenate([variables_in_rows, variables_in_cols]))
    return int(len(unique_vars))


def is_empty(qubo: QUBO) -> bool:
    return np.count_nonzero(qubo.mat) == 0 or qubo.problem_size == 0


def is_sparse(qubo: QUBO) -> bool:
    return vars_count(qubo) <= int(os.environ["CUT_DIM"])