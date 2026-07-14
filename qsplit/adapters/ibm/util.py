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


def get_variables_mapping(qubo: QUBO) -> tuple[dict[int, int], list[int]]:
    all_vars = sorted(list(set(qubo.rows_idx) | set(qubo.cols_idx)))
    var_to_qubit = {var: i for i, var in enumerate(all_vars)}
    return var_to_qubit, all_vars


def to_dataframe(
    counts_int: dict[int, int], qubo: QUBO, var_to_qubit: dict[int, int], all_vars: list[int]
) -> pd.DataFrame:
    data = []
    num_qubits = len(all_vars)

    valid_row_mask = [i for i, r in enumerate(qubo.rows_idx) if r != -1]
    valid_col_mask = [i for i, c in enumerate(qubo.cols_idx) if c != -1]

    valid_rows_idx = [qubo.rows_idx[i] for i in valid_row_mask]
    valid_cols_idx = [qubo.cols_idx[i] for i in valid_col_mask]

    mat_valid = qubo.mat[np.ix_(valid_row_mask, valid_col_mask)]

    for state_int, _ in counts_int.items():
        bin_str = np.binary_repr(state_int, width=num_qubits)
        full_solution = np.array([int(bit) for bit in bin_str])[::-1]

        sol_dict = {var_name: full_solution[q_idx] for var_name, q_idx in var_to_qubit.items()}

        vec_row = np.array([sol_dict[r] for r in valid_rows_idx])
        vec_col = np.array([sol_dict[c] for c in valid_cols_idx])
        energy = vec_row @ mat_valid @ vec_col.T

        if -1 in sol_dict:
            del sol_dict[-1]

        row = sol_dict.copy()
        row["energy"] = energy
        data.append(row)

    res = pd.DataFrame(data)
    res = res.sort_values(by="energy", ascending=True)

    cols = [c for c in res.columns if c != "energy"]
    cols.sort()
    res = res[cols + ["energy"]]

    best_energy = res["energy"].min()

    return res[res["energy"] == best_energy]
