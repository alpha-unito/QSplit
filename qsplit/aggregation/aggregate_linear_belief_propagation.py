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

from qsplit.qubo import QUBO


def aggregate_solutions(solutions: list[QUBO], qubo: QUBO) -> QUBO:
    all_indices = sorted([idx for idx in set(qubo.rows_idx).union(qubo.cols_idx) if idx >= 0])
    idx_to_pos = {idx: i for i, idx in enumerate(all_indices)}
    beliefs_1 = np.zeros(len(all_indices))
    beliefs_0 = np.zeros(len(all_indices))

    for sub_qubo in solutions:
        df = sub_qubo.solutions
        valid_columns = [col for col in df.columns if col in idx_to_pos]
        min_energy = df["energy"].min()
        energies = df["energy"].to_numpy()
        weights = np.exp(-(energies - min_energy))
        weights /= np.sum(weights)

        for idx_row, (_, row) in enumerate(df.iterrows()):
            w = weights[idx_row]
            for col in valid_columns:
                pos = idx_to_pos[col]
                val = row[col]
                beliefs_1[pos] += val * w
                beliefs_0[pos] += (1.0 - val) * w

    x = np.array([int(beliefs_1[i] >= beliefs_0[i]) for i in range(len(all_indices))])
    n = len(all_indices)
    energy = x.T @ qubo.mat[:n, :n] @ x
    sol_dict = {all_indices[i]: [x[i]] for i in range(len(all_indices))}
    sol_dict["energy"] = [float(energy)]
    qubo.solutions = pd.DataFrame(sol_dict)
    return qubo
