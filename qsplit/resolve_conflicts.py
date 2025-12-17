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


def local_search(df: pd.DataFrame, qubo: QUBO) -> pd.DataFrame:
    for i, row in df.iterrows():
        no_energy = row.drop('energy')
        var_num = len(no_energy)

        if not np.any(np.isnan(no_energy.values)):
            df.loc[i, 'energy'] = no_energy.values.T @ qubo.mat[:var_num, :var_num] @ no_energy.values
        else:
            nans = no_energy[np.isnan(no_energy)]
            idxs = np.array(nans.index.astype(int))
            qubo_nans = qubo.mat[np.ix_(idxs, idxs)]
            if np.count_nonzero(qubo_nans) == 0:
                nans_sol = solve_zeros(QUBO(qubo_nans, cols_idx=idxs, rows_idx=idxs))
            else:
                nans_sol = solve(QUBO(qubo_nans, cols_idx=idxs, rows_idx=idxs))
            best_sol = nans_sol.sort_values(by='energy', ascending=True).iloc[0]
            df.loc[i, idxs] = best_sol[idxs]
            df.loc[i, 'energy'] = df.iloc[i][:-1] @ qubo.mat[:var_num, :var_num] @ df.iloc[i][:-1].T

    return df
