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
import warnings

import numpy as np

from qsplit.adapters.dummy import solve as dummy_solve
from qsplit.adapters.dwave.dwave_sa import solve

# from qsplit.adapters.ibm.ibm_default import solve
from qsplit.aggregation.aggregate_k_interactions import aggregate_solutions as aggregate_solutions_interactions
from qsplit.aggregation.aggregate_linear import aggregate_solutions as aggregate_solutions_linear
from qsplit.aggregation.aggregate_linear_belief_propagation import aggregate_solutions as aggregate_solutions_linear_bp
from qsplit.aggregation.aggregate_recursive import aggregate_solutions as aggregate_solutions_recursive
from qsplit.aggregation.aggregate_recursive import aggregate_solutions_trivial
from qsplit.aggregation.aggregate_recursive_graph import aggregate_solutions as aggregate_solutions_recursive_graph
from qsplit.halting_heuristic.stop import is_empty, is_sparse
from qsplit.qubo import QUBO
from qsplit.splitting.split_k_interactions import split_problem as split_problem_interactions
from qsplit.splitting.split_linear import split_problem as split_problem_linear
from qsplit.splitting.split_recursive import split_problem as split_problem_recursive
from qsplit.splitting.split_recursive_graph import split_problem as split_problem_recursive_graph

warnings.warn(
    "local_runner module is deprecated. This was the legacy method for using QSplit. "
    "It is now recommended to use StreamFlow to leverage multiple quantum backends. "
    "For more information on using StreamFlow with QSplit, please refer to the README.md file.",
    DeprecationWarning,
    stacklevel=1,
)

LOGICAL_EXPANSION = False
BP = False


def logical_expansion(subs: tuple[QUBO, QUBO, QUBO]) -> tuple[QUBO, QUBO, QUBO]:
    ul, ur, lr = subs
    hints = __extract_logical_hints(ur)
    if not hints:
        return subs

    ul_diagonal = {idx: pos for pos, idx in enumerate(ul.rows_idx) if idx >= 0}
    lr_diagonal = {idx: pos for pos, idx in enumerate(lr.rows_idx) if idx >= 0}

    for row_pos, row_idx in enumerate(ur.rows_idx):
        for col_pos, col_idx in enumerate(ur.cols_idx):
            coefficient = ur.mat[row_pos, col_pos]
            if coefficient == 0:
                continue

            col_hint = hints.get(col_idx)
            if row_idx in ul_diagonal and col_hint is not None:
                ul_pos = ul_diagonal[row_idx]
                ul.mat[ul_pos, ul_pos] += coefficient * col_hint

            row_hint = hints.get(row_idx)
            if col_idx in lr_diagonal and row_hint is not None:
                lr_pos = lr_diagonal[col_idx]
                lr.mat[lr_pos, lr_pos] += coefficient * row_hint

    return subs


def __extract_logical_hints(qubo: QUBO) -> dict[int, float]:
    if qubo.solutions is None or qubo.solutions.empty or "energy" not in qubo.solutions.columns:
        return {}

    best_energy = qubo.solutions["energy"].min()
    best_solutions = qubo.solutions[qubo.solutions["energy"] == best_energy]
    hints = {}

    for col in best_solutions.columns:
        if col == "energy" or col < 0:
            continue

        values = best_solutions[col].replace([np.inf, -np.inf], np.nan).dropna()
        if not values.empty:
            hints[col] = float(values.mean())

    return hints


def qsplit_sampler_recursive(qubo: QUBO) -> QUBO:
    if is_empty(qubo):
        qubo.solutions = dummy_solve(qubo)
        return qubo
    if (qubo.problem_size <= int(os.environ["CUT_DIM"])) or is_sparse(qubo):
        qubo.solutions = solve(qubo)
        return qubo

    subs = split_problem_recursive(qubo)
    if LOGICAL_EXPANSION:
        subs[1].solutions = qsplit_sampler_recursive(subs[1]).solutions
        subs = logical_expansion(subs)
        subs[0].solutions = qsplit_sampler_recursive(subs[0]).solutions
        subs[2].solutions = qsplit_sampler_recursive(subs[2]).solutions
        return aggregate_solutions_trivial(subs[0], subs[2], qubo)
    else:
        subs[0].solutions = qsplit_sampler_recursive(subs[0]).solutions
        subs[1].solutions = qsplit_sampler_recursive(subs[1]).solutions
        subs[2].solutions = qsplit_sampler_recursive(subs[2]).solutions
        return aggregate_solutions_recursive(subs, qubo)


def qsplit_sampler_iterative(qubo: QUBO) -> QUBO:
    subs = split_problem_linear(qubo)
    for p in subs:
        if is_empty(p):
            p.solutions = dummy_solve(p)
        else:
            p.solutions = solve(p)
    return aggregate_solutions_linear_bp(subs, qubo) if BP else aggregate_solutions_linear(subs, qubo)


def qsplit_sampler_interactions(qubo: QUBO) -> QUBO:
    subs = split_problem_interactions(qubo)
    for p in subs:
        if is_empty(p):
            p.solutions = dummy_solve(p)
        else:
            p.solutions = solve(p)
    return aggregate_solutions_interactions(subs, qubo)


def qsplit_sampler_graph_partitioning(qubo: QUBO) -> QUBO:
    subs = split_problem_recursive_graph(qubo)
    for p in subs:
        if is_empty(p):
            p.solutions = dummy_solve(p)
        else:
            p.solutions = solve(p)
    return aggregate_solutions_recursive_graph(subs, qubo)
