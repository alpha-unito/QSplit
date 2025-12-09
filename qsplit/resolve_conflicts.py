import numpy as np
import pandas as pd

from qsplit.adapters.dwave.dwave_sa import solve
from qsplit.adapters.all_zero import solve as solve_zeros
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
