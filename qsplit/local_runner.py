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

from qsplit.halting_heuristic.stop import is_empty, is_sparse
from qsplit.adapters.dummy import solve as dummy_solve
from qsplit.adapters.dwave.dwave_sa import solve
from qsplit.aggregation.aggregate_linear import aggregate_solutions as aggregate_solutions_linear
from qsplit.aggregation.aggregate_recursive import aggregate_solutions as aggregate_solutions_recursive
from qsplit.qubo import QUBO
from qsplit.splitting.split_linear import split_problem as split_problem_linear
from qsplit.splitting.split_recursive import split_problem as split_problem_recursive

warnings.warn("local_runner module is deprecated. This was the legacy method for using QSplit. "
              "It is now recommended to use StreamFlow to leverage multiple quantum backends. "
              "For more information on using StreamFlow with QSplit, please refer to the README.md file.",
              DeprecationWarning, stacklevel=1)

LOGICAL_EXPANSION = False


def logical_expansion(subs: tuple[QUBO, QUBO, QUBO]) -> tuple[QUBO, QUBO, QUBO]:
    # TODO: Expand subs[1].solutions as hint in subs[0] and subs[2]
    return subs


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
    else:
        subs[0].solutions = qsplit_sampler_recursive(subs[0]).solutions
        subs[1].solutions = qsplit_sampler_recursive(subs[1]).solutions
        subs[2].solutions = qsplit_sampler_recursive(subs[2]).solutions
    return aggregate_solutions_recursive(subs, qubo)


def qsplit_sampler_iterative(qubo: QUBO) -> QUBO:
    subs = split_problem_linear(qubo)
    for p in subs:
        if is_empty(p) or is_sparse(qubo):
            p.solutions = dummy_solve(p)
        else:
            p.solutions = solve(p)
    return aggregate_solutions_linear(subs, qubo)
