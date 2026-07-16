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
import pymetis

from qsplit.qubo import QUBO


def _create_sub_qubo(qubo: QUBO, active_indices: list[int]) -> QUBO:
    indices_arr = np.array(active_indices, dtype=int)
    sub_mat = qubo.mat[np.ix_(indices_arr, indices_arr)].copy()
    sub_rows = qubo.rows_idx[indices_arr].copy()
    sub_cols = qubo.cols_idx[indices_arr].copy()
    return QUBO(sub_mat, sub_rows, sub_cols, offset=qubo.offset)


def split_problem(qubo: QUBO) -> list[QUBO]:
    cut_dim = min(int(os.environ.get("CUT_DIM")), qubo.problem_size)
    real_nodes = [i for i in range(qubo.problem_size) if qubo.cols_idx[i] != -1]
    size = qubo.problem_size
    adjacency = [[] for _ in range(size)]

    for i in range(size):
        if qubo.cols_idx[i] == -1:
            continue
        for j in range(i + 1, size):
            if qubo.cols_idx[j] == -1:
                continue
            if qubo.mat[i, j] != 0.0:
                adjacency[i].append(j)
                adjacency[j].append(i)

    all_partitions = __pymetis_split(adjacency, real_nodes, cut_dim)
    return [_create_sub_qubo(qubo, partition) for partition in all_partitions]


def __pymetis_split(adjacency: list[list[int]], current_nodes: list[int], max_size: int) -> list[list[int]]:
    if len(current_nodes) <= max_size:
        return [current_nodes]

    sub_node_set = set(current_nodes)
    sub_node_list = list(current_nodes)
    loc_to_glo = {i: node for i, node in enumerate(sub_node_list)}
    glo_to_loc = {node: i for i, node in enumerate(sub_node_list)}

    local_adjacency = [[] for _ in range(len(sub_node_list))]
    has_edges = False
    for local_idx, global_node in enumerate(sub_node_list):
        for neighbor in adjacency[global_node]:
            if neighbor in sub_node_set:
                local_adjacency[local_idx].append(glo_to_loc[neighbor])
                has_edges = True

    if not has_edges:
        half = len(sub_node_list) // 2
        part1 = sub_node_list[:half]
        part2 = sub_node_list[half:]
    else:
        _, member = pymetis.part_graph(nparts=2, adjacency=local_adjacency)
        part1 = [loc_to_glo[i] for i, part in enumerate(member) if part == 0]
        part2 = [loc_to_glo[i] for i, part in enumerate(member) if part == 1]

    return __pymetis_split(adjacency, part1, max_size) + __pymetis_split(adjacency, part2, max_size)
