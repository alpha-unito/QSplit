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

from itertools import product

import numpy as np
import pandas as pd

from qsplit.adapters.all_zero import solve as solve_zeros
from qsplit.adapters.dwave.dwave_sa import solve
from qsplit.qubo import QUBO

EXACT_CONFLICT_LIMIT = 10


def calculate_energy(x, q_mat):
    x_clean = np.where(np.isinf(x), 0.0, x)
    return x_clean.T @ q_mat @ x_clean


def nan_subqubo(df: pd.DataFrame, qubo: QUBO) -> pd.DataFrame:
    df = df.copy()
    for row_idx, row in df.iterrows():
        columns_by_variable = __columns_by_variable(row)
        missing_variables = __missing_variables(row, columns_by_variable)
        if missing_variables:
            best_sol = __solve_missing_variables(row, columns_by_variable, missing_variables, qubo)
            for variable in missing_variables:
                target_col = columns_by_variable[variable]
                df.at[row_idx, target_col] = best_sol.get(variable, 0.0)

        df.loc[row_idx, "energy"] = __calculate_qubo_energy(df.loc[row_idx], qubo)

    return df


def __columns_by_variable(row: pd.Series) -> dict[int, object]:
    columns_by_variable = {}
    for col in row.index:
        if col == "energy":
            continue

        variable = int(col)
        if variable is not None and variable >= 0:
            columns_by_variable[variable] = col

    return columns_by_variable


def __missing_variables(row: pd.Series, columns_by_variable: dict[int, object]) -> list[int]:
    return sorted([variable for variable, col in columns_by_variable.items() if pd.isna(row[col])])


def __solve_missing_variables(
    row: pd.Series,
    columns_by_variable: dict[int, object],
    missing_variables: list[int],
    qubo: QUBO,
) -> dict[int, float]:
    local_qubo = __build_local_qubo(row, columns_by_variable, missing_variables, qubo)
    if local_qubo.problem_size == 0 or np.count_nonzero(local_qubo.mat) == 0:
        return dict.fromkeys(missing_variables, 0.0)

    if len(missing_variables) <= EXACT_CONFLICT_LIMIT:
        return __solve_exact(local_qubo)

    nans_sol = __solve(local_qubo)
    best_sol = nans_sol.sort_values(by="energy").iloc[0]
    return {
        variable: float(best_sol[variable]) if variable in best_sol.index else 0.0 for variable in missing_variables
    }


def __build_local_qubo(
    row: pd.Series,
    columns_by_variable: dict[int, object],
    missing_variables: list[int],
    qubo: QUBO,
) -> QUBO:
    variable_to_pos = {variable: pos for pos, variable in enumerate(missing_variables)}
    local_mat = np.zeros((len(missing_variables), len(missing_variables)))

    for row_pos, row_variable in enumerate(qubo.rows_idx):
        if row_variable < 0:
            continue

        for col_pos, col_variable in enumerate(qubo.cols_idx):
            if col_variable < 0:
                continue

            coefficient = qubo.mat[row_pos, col_pos]
            if coefficient == 0:
                continue

            row_missing = row_variable in variable_to_pos
            col_missing = col_variable in variable_to_pos
            if row_missing and col_missing:
                local_mat[variable_to_pos[row_variable], variable_to_pos[col_variable]] += coefficient
            elif row_missing:
                fixed_col_value = __row_value(row, columns_by_variable, col_variable)
                local_mat[variable_to_pos[row_variable], variable_to_pos[row_variable]] += coefficient * fixed_col_value
            elif col_missing:
                fixed_row_value = __row_value(row, columns_by_variable, row_variable)
                local_mat[variable_to_pos[col_variable], variable_to_pos[col_variable]] += coefficient * fixed_row_value

    indices = np.array(missing_variables)
    return QUBO(local_mat, rows_idx=indices, cols_idx=indices)


def __solve_exact(qubo: QUBO) -> dict[int, float]:
    variables = sorted([idx for idx in set(qubo.rows_idx).union(qubo.cols_idx) if idx >= 0])
    best_assignment = dict.fromkeys(variables, 0.0)
    best_energy = np.inf

    for values in product([0.0, 1.0], repeat=len(variables)):
        assignment = dict(zip(variables, values))
        row_values = np.array([assignment.get(idx, 0.0) for idx in qubo.rows_idx])
        col_values = np.array([assignment.get(idx, 0.0) for idx in qubo.cols_idx])
        energy = float(row_values.T @ qubo.mat @ col_values)
        if energy < best_energy:
            best_energy = energy
            best_assignment = assignment

    return best_assignment


def __calculate_qubo_energy(row: pd.Series, qubo: QUBO) -> float:
    columns_by_variable = __columns_by_variable(row)
    row_values = np.array([__row_value(row, columns_by_variable, idx) for idx in qubo.rows_idx])
    col_values = np.array([__row_value(row, columns_by_variable, idx) for idx in qubo.cols_idx])
    return float(row_values.T @ qubo.mat @ col_values)


def __row_value(row: pd.Series, columns_by_variable: dict[int, object], variable: int) -> float:
    if variable < 0 or variable not in columns_by_variable:
        return 0.0

    value = row[columns_by_variable[variable]]
    if pd.isna(value) or np.isinf(value):
        return 0.0
    return float(value)


def __solve(qubo: QUBO) -> pd.DataFrame:
    if np.count_nonzero(qubo.mat) == 0:
        return solve_zeros(qubo)

    return solve(qubo)
