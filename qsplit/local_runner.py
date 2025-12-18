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

import warnings

import numpy as np

from qsplit.adapters.dummy import solve as dummy_solve
from qsplit.adapters.dwave.dwave_sa import solve
from qsplit.aggregation.aggregate_recursive import aggregate_solutions
from qsplit.qubo import QUBO
from qsplit.splitting.split_recursive import split_problem


def qsplit_sampler(qubo: QUBO, cut_dim: int) -> QUBO:
    warnings.warn("qsplit_sampler is deprecated. This was the legacy method for using QSplit. "
                  "It is now recommended to use StreamFlow to leverage multiple quantum backends. "
                  "For more information on using StreamFlow with QSplit, please refer to the README.md file.",
                  DeprecationWarning, stacklevel=2)

    if np.count_nonzero(qubo.mat) == 0 or qubo.problem_size == 0:
        qubo.solutions = dummy_solve(qubo)
        return qubo
    if qubo.problem_size <= cut_dim:
        qubo.solutions = solve(qubo)
        return qubo

    subs = split_problem(qubo)
    sols = [qsplit_sampler(p, cut_dim) for p in subs]
    return aggregate_solutions((sols[0], sols[1], sols[2]), qubo)
