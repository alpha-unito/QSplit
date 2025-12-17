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

import unittest

import numpy as np

from qsplit.qubo import QUBO
from qsplit.split import split_problem


class TestSplitProblem(unittest.TestCase):
    def test_split_even_dimension(self):
        dim = 4
        mat = np.zeros((dim, dim))
        mat[0, 0] = 9.0
        mat[0, 2] = 5.0
        mat[2, 2] = 2.0
        cols = np.array([0, 1, 2, 3])
        rows = np.array([0, 1, 2, 3])

        qubo = QUBO(mat, cols_idx=cols, rows_idx=rows)
        res_ul, res_ur, res_lr = split_problem(qubo)

        # UL check
        self.assertEqual(res_ul.mat[0, 0], 9.0)
        self.assertEqual(np.count_nonzero(res_ul), 1)
        self.assertEqual(res_ul.cols_idx.tolist(), [0, 1])

        # UR check
        self.assertEqual(res_ur.mat[0, 0], 5.0)
        self.assertEqual(np.count_nonzero(res_ur.mat), 1)
        self.assertEqual(res_ur.cols_idx.tolist(), [2, 3])
        self.assertEqual(res_ur.rows_idx.tolist(), [0, 1])

        # LR check
        self.assertEqual(res_lr.mat[0, 0], 2.0)
        self.assertEqual(np.count_nonzero(res_lr.mat), 1)
        self.assertEqual(res_lr.rows_idx.tolist(), [2, 3])

    def test_split_odd_dimension(self):
        dim = 5
        mat = np.zeros((dim, dim))
        mat[0, 0] = 1.0
        mat[0, 3] = 2.0
        mat[3, 3] = 3.0
        qubo = QUBO(mat, np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 4]))

        res_ul, res_ur, res_lr = split_problem(qubo)

        # UL check
        self.assertEqual(res_ul.mat[0, 0], 1.0)
        self.assertEqual(np.count_nonzero(res_ul), 1)
        self.assertEqual(res_ul.cols_idx.tolist(), [0, 1, 2, -1])

        # UR check
        print(res_ur)
        self.assertEqual(res_ur.mat[0, 0], 2.0)
        self.assertEqual(np.count_nonzero(res_ur.mat), 1)
        self.assertEqual(res_ur.cols_idx.tolist(), [3, 4, -1, -1])
        self.assertEqual(res_ur.rows_idx.tolist(), [0, 1, 2, -1])

        # LR check
        self.assertEqual(res_lr.mat[0, 0], 3.0)
        self.assertEqual(np.count_nonzero(res_lr.mat), 1)
        self.assertEqual(res_lr.rows_idx.tolist(), [3, 4, -1, -1])

    def test_raise_value_error_for_lower_left(self):
        dim = 4
        qubo = QUBO(np.zeros((dim, dim)), np.array([0, 1, 2, 3]), np.array([1, 2, 3, 4]))
        qubo.mat[2, 0] = 99.0

        with self.assertRaises(ValueError) as context:
            split_problem(qubo)
        self.assertIn("Lower left sub-matrix must be 0", str(context.exception))

    def test_indices_slicing(self):
        dim = 6
        cols = np.array([0, 1, 2, 3, 4, 5])
        rows = np.array([0, 1, 2, 3, 4, 5])
        qubo = QUBO(np.zeros((dim, dim)), rows_idx=rows, cols_idx=cols)
        ul, ur, lr = split_problem(qubo)

        self.assertEqual(ul.cols_idx.tolist(), [0, 1, 2, -1])
        self.assertEqual(ul.rows_idx.tolist(), [0, 1, 2, -1])

        self.assertEqual(ur.cols_idx.tolist(), [3, 4, 5, -1])
        self.assertEqual(ur.rows_idx.tolist(), [0, 1, 2, -1])

        self.assertEqual(lr.cols_idx.tolist(), [3, 4, 5, -1])
        self.assertEqual(lr.rows_idx.tolist(), [3, 4, 5, -1])


if __name__ == '__main__':
    unittest.main()
