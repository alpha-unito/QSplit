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

import pandas as pd

from qsplit.qubo import QUBO


def aggregate_solutions(solutions: list[QUBO], qubo: QUBO) -> QUBO:
    all_indices = sorted(list(set(qubo.rows_idx).union(qubo.cols_idx)))
    idx_to_pos = {idx: i for i, idx in enumerate(all_indices)}
    out_solutions = [0.5] * len(all_indices)
    counts = [0] * len(all_indices)

    for sub_qubo in solutions:
        df = sub_qubo.solutions
        valid_columns = [col for col in df.columns if col in idx_to_pos]

        for _, row in df.iterrows():
            for col in valid_columns:
                pos = idx_to_pos[col]
                out_solutions[pos] += row[col]
                counts[pos] += 1

    for i in range(len(out_solutions)):
        if counts[i] > 0:
            out_solutions[i] = out_solutions[i] / counts[i]

    sol_dict = {all_indices[i]: [round(out_solutions[i])] for i in range(len(all_indices))}
    qubo.solutions = pd.DataFrame(sol_dict)

    return qubo
