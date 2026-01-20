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
import pandas as pd

from qsplit.adapters.all_zero import solve as zero_solve
from qsplit.adapters.dummy import solve as dummy_solve
from qsplit.qubo import QUBO


class TestSolveNan(unittest.TestCase):
    def test_solve_nan_even_size(self):
        mat = np.array([[1.0, 2.0], [0.0, 3.0]])
        rows_idx = np.array([0, 1])
        cols_idx = np.array([0, 1])
        qubo = QUBO(mat, rows_idx, cols_idx)
        result_df = dummy_solve(qubo)

        self.assertTrue(isinstance(result_df, pd.DataFrame))
        self.assertEqual(result_df.shape, (1, 3))
        self.assertEqual(list(result_df.columns), [0, 1, "energy"])
        self.assertTrue(np.isnan(result_df.loc[0, 0]))
        self.assertTrue(np.isnan(result_df.loc[0, 1]))
        self.assertTrue(np.isnan(result_df.loc[0, "energy"]))

    def test_solve_nan_odd_size_sanitized(self):
        mat = np.array([[5.0]])
        rows_idx = np.array([10])
        cols_idx = np.array([10])
        qubo = QUBO(mat, rows_idx, cols_idx)
        result_df = dummy_solve(qubo)

        self.assertEqual(result_df.shape, (1, 3))
        self.assertEqual(list(result_df.columns), [-1, 10, "energy"])
        self.assertTrue(np.isnan(result_df.loc[0, -1]))
        self.assertTrue(np.isnan(result_df.loc[0, 10]))
        self.assertTrue(np.isnan(result_df.loc[0, "energy"]))

    def test_solve_nan_non_contiguous_indices(self):
        mat = np.array([[1.0, 2.0], [0.0, 3.0]])
        rows_idx = np.array([5, 10])
        cols_idx = np.array([5, 10])
        qubo = QUBO(mat, rows_idx, cols_idx)
        result_df = dummy_solve(qubo)

        self.assertEqual(list(result_df.columns), [5, 10, "energy"])
        self.assertTrue(np.all(np.isnan(result_df.values)))


class TestSolveZero(unittest.TestCase):
    def test_solve_zero_even_size(self):
        mat = np.array([[1.0, 2.0], [0.0, 3.0]])
        rows_idx = np.array([0, 1])
        cols_idx = np.array([0, 1])
        qubo = QUBO(mat, rows_idx, cols_idx)
        result_df = zero_solve(qubo)

        self.assertTrue(isinstance(result_df, pd.DataFrame))
        self.assertEqual(result_df.shape, (1, 3))
        self.assertEqual(list(result_df.columns), [0, 1, "energy"])
        self.assertEqual(result_df.loc[0, 0], 0)
        self.assertEqual(result_df.loc[0, 1], 0)
        self.assertEqual(result_df.loc[0, "energy"], 0)
        self.assertTrue(np.all(result_df.drop(columns=["energy"]).values == 0))

    def test_solve_zero_odd_size_sanitized(self):
        mat = np.diag([1.0, 2.0, 3.0])
        rows_idx = np.array([1, 2, 3])
        cols_idx = np.array([1, 2, 3])
        qubo = QUBO(mat, rows_idx, cols_idx)
        result_df = zero_solve(qubo)

        self.assertEqual(result_df.shape, (1, 5))
        self.assertEqual(list(result_df.columns), [-1, 1, 2, 3, "energy"])
        self.assertTrue(np.all(result_df.values == 0))


if __name__ == "__main__":
    unittest.main()
