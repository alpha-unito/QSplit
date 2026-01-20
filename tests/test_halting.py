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
import unittest

import numpy as np

from qsplit.halting_heuristic.stop import is_empty, is_sparse
from qsplit.qubo import QUBO


class TestHalting(unittest.TestCase):
    def setUp(self):
        self.cut_dim = 5

    def test_is_empty_with_zeros(self):
        q = QUBO(mat=np.zeros((3, 3)), rows_idx=np.array([0, 1, 2]), cols_idx=np.array([0, 1, 2]))
        self.assertTrue(is_empty(q))

    def test_is_not_empty(self):
        mat = np.zeros((2, 2))
        mat[0, 1] = 1.5
        q = QUBO(mat=mat, rows_idx=np.array([0, 1]), cols_idx=np.array([0, 1]))
        self.assertFalse(is_empty(q))

    def test_is_sparse_within_limit(self):
        os.environ["CUT_DIM"] = str(5)
        mat = np.zeros((2, 2))
        mat[0, 0] = 1
        mat[0, 1] = 1
        mat[1, 1] = 1
        q = QUBO(mat=mat, rows_idx=np.array([10, 20]), cols_idx=np.array([10, 30]))

        self.assertTrue(is_sparse(q))

    def test_is_sparse_exceeds_limit(self):
        os.environ["CUT_DIM"] = str(2)
        mat = np.array([[1, 1], [0, 1]])
        q = QUBO(mat=mat, rows_idx=np.array([1, 2]), cols_idx=np.array([1, 3]))

        self.assertFalse(is_sparse(q))

    def test_is_sparse_with_duplicate_mappings(self):
        os.environ["CUT_DIM"] = str(5)
        mat = np.ones((2, 2))
        q = QUBO(mat=mat, rows_idx=np.array([100, 100]), cols_idx=np.array([100, 100]))

        self.assertTrue(is_sparse(q))


if __name__ == "__main__":
    unittest.main()
