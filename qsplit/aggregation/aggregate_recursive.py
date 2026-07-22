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

from qsplit.conflict_resolution.local_search import nan_subqubo
from qsplit.qubo import QUBO


def aggregate_solutions_trivial(ul: QUBO, lr: QUBO, qubo: QUBO) -> QUBO:
    combined_df = __combine_ul_lr(ul, lr)
    qubo.solutions = __keep_min_energy_solutions(__recalculate_energy(combined_df, qubo), qubo)
    return qubo


def aggregate_solutions(solutions: tuple[QUBO, QUBO, QUBO], qubo: QUBO) -> QUBO:
    # Aggregate upper-left qubo with lower-right
    starting_sols = __combine_ul_lr(solutions[0], solutions[2])
    if -1 in starting_sols.columns:
        starting_sols[-1] = 0
    # Set missing columns in upper-right qubo to np.inf
    ur_qubo_filled = __fill_with_inf(starting_sols.columns, solutions[1].solutions)
    # Search the closest assignments between upper-right qubo and merged solution (UL and LR qubos)
    closest_df = __get_closest_assignments(starting_sols, ur_qubo_filled)

    # Combine
    combined_df = pd.DataFrame(
        [__combine_rows(row1, row2) for (_, row1), (_, row2) in zip(starting_sols.iterrows(), closest_df.iterrows())],
        columns=starting_sols.columns,
    )

    # Conflicts resolution
    qubo.solutions = (
        nan_subqubo(combined_df, qubo).reset_index(drop=True).drop_duplicates().nsmallest(n=10, columns="energy")
    )

    return qubo


def __combine_rows(row1: pd.Series, row2: pd.Series) -> list[float]:
    combined_row = []
    for col in row1.index:
        val1, val2 = row1[col], row2[col]
        if col == "energy":
            has_conflict = any(np.isnan(x) for x in combined_row)

            if has_conflict or (np.isnan(val1) and np.isnan(val2)):
                combined_row.append(np.nan)
            elif np.isnan(val1):
                combined_row.append(val2)
            elif np.isnan(val2):
                combined_row.append(val1)
            else:
                combined_row.append(val1 + val2)
        else:
            if np.isnan(val1) or np.isnan(val2):
                combined_row.append(np.nan)
            elif np.isinf(val1) and np.isinf(val2):
                combined_row.append(np.inf)
            elif np.isinf(val1):
                combined_row.append(val2)
            elif np.isinf(val2):
                combined_row.append(val1)
            elif val1 == val2:
                combined_row.append(val1)
            else:
                combined_row.append(np.nan)
    return combined_row


def __distance(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) == 0:
        return np.inf
    return np.sum(a[mask] != b[mask]) / np.sum(mask)


def __get_closest_assignments(starting_sols: pd.DataFrame, ur_qubo_filled: pd.DataFrame) -> pd.DataFrame:
    closest_rows = []
    for _, row in starting_sols.iterrows():
        distances = []
        for _, sol_row in ur_qubo_filled.iterrows():
            distance = __distance(row.values[:-1], sol_row.values[:-1])
            distances.append(distance)
        closest_idx = np.argmin(distances)
        to_append = ur_qubo_filled.iloc[closest_idx].copy()
        if np.any(to_append.isna()):
            to_append["energy"] = np.nan
        closest_rows.append(to_append)
    return pd.DataFrame(closest_rows).reset_index(drop=True)


def __fill_with_inf(schema: pd.Index, df_to_fill: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [col for col in schema if col not in df_to_fill.columns]
    if missing_columns:
        df_missing = pd.DataFrame(np.inf, index=df_to_fill.index, columns=missing_columns)
        df_to_fill = pd.concat([df_to_fill, df_missing], axis=1)
    return df_to_fill[schema]


def __combine_ul_lr(ul: QUBO, lr: QUBO) -> pd.DataFrame:
    all_indices = sorted(list(set(ul.rows_idx).union(lr.cols_idx)))
    ul.solutions["tmp"] = 1
    lr.solutions["tmp"] = 1
    merge = pd.merge(ul.solutions, lr.solutions, on="tmp")
    merge["energy"] = merge["energy_x"] + merge["energy_y"]
    merge = merge.drop(["energy_x", "energy_y", "tmp"], axis=1)
    ul.solutions.drop("tmp", axis=1, inplace=True)
    lr.solutions.drop("tmp", axis=1, inplace=True)
    return __fill_with_inf(pd.Index(all_indices + ["energy"]), merge)


def __recalculate_energy(df: pd.DataFrame, qubo: QUBO) -> pd.DataFrame:
    real_indices = sorted([idx for idx in set(qubo.rows_idx).union(qubo.cols_idx) if idx >= 0])
    df = __fill_with_inf(pd.Index(real_indices + ["energy"]), df)

    for row_idx, row in df.iterrows():
        assignment = {idx: np.nan_to_num(row[idx], nan=0.0, posinf=0.0, neginf=0.0) for idx in real_indices}
        row_values = np.array([assignment.get(idx, 0.0) for idx in qubo.rows_idx])
        col_values = np.array([assignment.get(idx, 0.0) for idx in qubo.cols_idx])
        df.loc[row_idx, "energy"] = float(row_values.T @ qubo.mat @ col_values)

    return df


def __keep_min_energy_solutions(df: pd.DataFrame, qubo: QUBO) -> pd.DataFrame:
    real_columns = sorted([idx for idx in set(qubo.rows_idx).union(qubo.cols_idx) if idx >= 0])
    df = df[real_columns + ["energy"]].reset_index(drop=True).drop_duplicates()
    if df.empty:
        return df

    min_energy = df["energy"].min()
    return df[df["energy"] == min_energy].reset_index(drop=True)
