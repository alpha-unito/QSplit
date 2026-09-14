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

import os

import numpy as np

from qsplit.qubo import QUBO


def split_problem(qubo: QUBO) -> list[QUBO]:
    cut_dim = int(os.environ["CUT_DIM"])
    cut_dim -= cut_dim % 2
    exact_ratio = float(os.environ.get("EXACT_RATIO", "0.75"))

    real_positions = np.flatnonzero(qubo.rows_idx >= 0)
    real_ids = qubo.rows_idx[real_positions]
    mat = qubo.mat[np.ix_(real_positions, real_positions)]
    if len(real_ids) <= cut_dim:
        return [QUBO(mat.copy(), real_ids.copy(), real_ids.copy(), offset=qubo.offset)]

    num_exact = max(1, int(cut_dim * exact_ratio))
    num_macro = cut_dim - num_exact
    mat_sym = mat + mat.T - np.diag(np.diag(mat))
    mat_abs = np.abs(mat_sym)
    np.fill_diagonal(mat_abs, 0.0)
    res = []

    for i in range(len(real_ids)):
        sorted_indices = np.argsort(-mat_abs[i], kind="stable")
        sorted_indices = sorted_indices[sorted_indices != i]
        exact_indices = [i] + sorted_indices[: num_exact - 1].tolist()
        remaining_indices = sorted_indices[num_exact - 1 :]
        macro_clusters = []
        if num_macro > 0:
            macro_clusters = [c.tolist() for c in np.array_split(remaining_indices, num_macro)]

        dimension = len(exact_indices) + len(macro_clusters)
        new_mat_sym = np.zeros((dimension, dimension))
        new_mat_sym[:num_exact, :num_exact] = mat_sym[np.ix_(exact_indices, exact_indices)]

        for m_idx, cluster in enumerate(macro_clusters):
            macro_pos = num_exact + m_idx
            for e_pos, e in enumerate(exact_indices):
                interaction = np.sum(mat_sym[e, cluster])
                new_mat_sym[e_pos, macro_pos] = interaction
                new_mat_sym[macro_pos, e_pos] = interaction

            c_mesh = np.ix_(cluster, cluster)
            internal_energy = (np.sum(mat_sym[c_mesh]) + np.sum(np.diag(mat_sym)[cluster])) / 2.0
            new_mat_sym[macro_pos, macro_pos] = internal_energy

            for m_idx2 in range(m_idx + 1, len(macro_clusters)):
                cluster2 = macro_clusters[m_idx2]
                macro_pos2 = num_exact + m_idx2
                cross_int = np.sum(mat_sym[np.ix_(cluster, cluster2)])
                new_mat_sym[macro_pos, macro_pos2] = cross_int
                new_mat_sym[macro_pos2, macro_pos] = cross_int

        sub_ids = np.array(real_ids[exact_indices].tolist() + [-(m + 1000) for m in range(num_macro)], dtype=int)
        res.append(QUBO(np.triu(new_mat_sym), sub_ids.copy(), sub_ids.copy(), offset=qubo.offset))

    return res
