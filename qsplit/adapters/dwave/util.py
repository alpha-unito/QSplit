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

import dimod
import numpy as np
import pandas as pd
from dimod import BinaryQuadraticModel, SampleSet

from qsplit.qubo import QUBO


def from_qubo_matrix_to_bqm(qubo: QUBO) -> BinaryQuadraticModel:
    rows_idx = np.array(qubo.rows_idx)
    cols_idx = np.array(qubo.cols_idx)
    mask = (rows_idx != -1) & (cols_idx != -1)
    filtered_mat = qubo.mat[np.ix_(mask, mask)]
    linear = np.diag(filtered_mat).astype(np.float64)
    q_rows, q_cols = np.triu_indices_from(filtered_mat, k=1)
    quad_values = filtered_mat[q_rows, q_cols].astype(np.float64)
    mask_quad = quad_values != 0
    quadratic = (q_rows[mask_quad], q_cols[mask_quad], quad_values[mask_quad])
    return BinaryQuadraticModel.from_numpy_vectors(linear, quadratic, offset=qubo.offset, vartype=dimod.BINARY)


def to_dataframe(sampleset: SampleSet, qubo: QUBO) -> pd.DataFrame:
    res = sampleset.to_pandas_dataframe()
    rename_map = {i: name for i, name in enumerate(qubo.cols_idx)}
    res.rename(columns=rename_map, inplace=True)
    res = res.drop(columns=["num_occurrences"]).drop_duplicates().sort_values(by="energy", ascending=True)
    return res[res["energy"] == min(res["energy"])]
