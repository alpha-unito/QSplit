from typing import List, Any

import numpy as np
import pandas as pd

from qsplit.qubo import QUBO
from qsplit.resolve_conflicts import local_search


def aggregate_solutions(solutions: List[QUBO], qubo: QUBO) -> QUBO:
    # Aggregate upper-left qubo with lower-right
    starting_sols = __combine_ul_lr(solutions[0], solutions[2])
    # Set missing columns in upper-right qubo to NaN
    ur_qubo_filled = __fill_with_nan(starting_sols.columns, solutions[1].solutions)
    # Search the closest assignments between upper-right qubo and merged solution (UL and LR qubos)
    closest_df = __get_closest_assignments(starting_sols, ur_qubo_filled)

    # Combine
    combined_df = pd.DataFrame(
        [__combine_rows(row1, row2) for (_, row1), (_, row2) in zip(starting_sols.iterrows(), closest_df.iterrows())],
        columns=starting_sols.columns)

    # Conflicts resolution
    qubo.solutions = (
        local_search(combined_df, qubo).reset_index(drop=True).drop_duplicates().nsmallest(n=10, columns='energy'))

    return qubo


def __combine_rows(row1: pd.Series, row2: pd.Series) -> List[float | Any]:
    combined_row = []
    for col in row1.index:
        val1, val2 = row1[col], row2[col]
        if col == 'energy':
            if (np.nan in combined_row) or (np.isnan(val1) and np.isnan(val2)):
                combined_row.append(np.nan)
            elif np.isnan(val1):
                combined_row.append(val2)
            elif np.isnan(val2):
                combined_row.append(val1)
            else:
                combined_row.append(val1 + val2)
        else:
            if pd.isna(val2) and not pd.isna(val1):
                combined_row.append(val1)
            elif pd.isna(val1) and not pd.isna(val2):
                combined_row.append(val2)
            elif val1 == val2:
                combined_row.append(val1)
            else:
                combined_row.append(np.nan)
    return combined_row


def __nan_hamming_distance(a: np.ndarray, b: np.ndarray) -> float | Any:
    mask = ~np.isnan(a) & ~np.isnan(b)
    if np.sum(mask) == 0:
        return np.inf
    return np.sum(a[mask] != b[mask]) / np.sum(mask)


def __get_closest_assignments(starting_sols: pd.DataFrame, ur_qubo_filled: pd.DataFrame) -> pd.DataFrame:
    closest_rows = []
    for i, row in starting_sols.iterrows():
        distances = []
        for j, sol_row in ur_qubo_filled.iterrows():
            distance = __nan_hamming_distance(row.values, sol_row.values)
            distances.append(distance)
        closest_idx = np.argmin(distances)
        to_append = ur_qubo_filled.iloc[closest_idx].copy()
        if np.any(to_append.isna()):
            to_append['energy'] = np.nan
        closest_rows.append(to_append)
    return pd.DataFrame(closest_rows).reset_index(drop=True)


def __fill_with_nan(schema: pd.Index, df_to_fill: pd.DataFrame) -> pd.DataFrame:
    missing_columns = set(schema) - set(df_to_fill.columns)
    for col in missing_columns:
        df_to_fill[col] = np.nan
    return df_to_fill[schema]


def __combine_ul_lr(ul: QUBO, lr: QUBO) -> pd.DataFrame:
    all_indices = sorted(list(set(ul.rows_idx).union(lr.cols_idx)))
    ul.solutions['tmp'] = 1
    lr.solutions['tmp'] = 1
    merge = pd.merge(ul.solutions, lr.solutions, on='tmp')
    merge['energy'] = merge['energy_x'] + merge['energy_y']
    merge = merge.drop(['energy_x', 'energy_y', 'tmp'], axis=1)
    ul.solutions.drop('tmp', axis=1, inplace=True)
    lr.solutions.drop('tmp', axis=1, inplace=True)
    return __fill_with_nan(pd.Index(all_indices + ['energy']), merge)
