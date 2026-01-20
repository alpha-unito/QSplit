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

from qsplit.adapters.all_zero import solve as solve_zeros
from qsplit.adapters.dwave.dwave_sa import solve
from qsplit.qubo import QUBO


def nan_subqubo(df: pd.DataFrame, qubo: QUBO) -> pd.DataFrame:
    for i, row in df.iterrows():
        no_energy = row.drop("energy")
        var_num = len(no_energy)

        if not np.any(np.isnan(no_energy.values)):
            df.loc[i, "energy"] = no_energy.values.T @ qubo.mat[:var_num, :var_num] @ no_energy.values
        else:
            nan_indices = no_energy[no_energy.isna()].index.astype(int)
            nan_rows = [idx for idx in nan_indices if idx in qubo.rows_idx]
            nan_cols = [idx for idx in nan_indices if idx in qubo.cols_idx]
            row_map = [list(qubo.rows_idx).index(r) for r in nan_rows]
            col_map = [list(qubo.cols_idx).index(c) for c in nan_cols]
            qubo_rect = qubo.mat[np.ix_(row_map, col_map)]
            size_r = len(nan_rows)
            size_c = len(nan_cols)
            n = max(size_r, size_c)
            qubo_square = np.zeros((n, n))
            qubo_square[:size_r, :size_c] = qubo_rect
            rows_padded = np.array(nan_rows + [-1] * (n - size_r))
            cols_padded = np.array(nan_cols + [-1] * (n - size_c))
            solver = solve_zeros if np.count_nonzero(qubo_square) == 0 else solve
            qubo_nan = QUBO(qubo_square, cols_idx=cols_padded, rows_idx=rows_padded)
            nans_sol = solver(qubo_nan)
            best_sol = nans_sol.sort_values(by="energy").iloc[0]
            for idx in nan_indices:
                if idx != -1:
                    target_col = None
                    if idx in df.columns:
                        target_col = idx
                    elif str(idx) in df.columns:
                        target_col = str(idx)
                    if target_col is not None:
                        val = 0.0
                        if target_col in best_sol.index:
                            val = best_sol[target_col]
                        df.at[i, target_col] = val
            full_row_values = df.iloc[i].drop("energy").values
            df.loc[i, "energy"] = full_row_values.T @ qubo.mat[:var_num, :var_num] @ full_row_values

    return df
