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

from collections import Counter

import numpy as np
import pandas as pd

from qsplit.qubo import QUBO


def aggregate_solutions(solutions: list[QUBO], qubo: QUBO) -> QUBO:
    all_indices = sorted([idx for idx in set(qubo.rows_idx) if idx >= 0])
    votes = {idx: [] for idx in all_indices}

    for sub_qubo in solutions:
        df_sol = sub_qubo.solutions
        best_sol = df_sol.nsmallest(1, "energy").iloc[0]
        for col in df_sol.columns:
            if col == "energy" or col < 0:
                continue
            elif col in votes:
                votes[col].append(best_sol[col])

    assignment = {}
    for idx in all_indices:
        var_votes = votes[idx]
        if len(var_votes) > 0:
            most_common_val, _ = Counter(var_votes).most_common(1)[0]
            assignment[idx] = int(round(most_common_val))
        else:
            assignment[idx] = 0

    x = np.array([assignment[idx] for idx in all_indices])
    n = len(x)
    energy = x.T @ qubo.mat[:n, :n] @ x
    sol_dict = {idx: [assignment[idx]] for idx in all_indices}
    sol_dict["energy"] = [float(energy)]
    qubo.solutions = pd.DataFrame(sol_dict)
    return qubo
