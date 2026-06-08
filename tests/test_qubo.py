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


class TestQUBO(unittest.TestCase):
    def setUp(self):
        self.valid_mat = np.array([[1.0, 2.0], [0.0, 3.0]])
        self.valid_rows_idx = np.array([1, 2])
        self.valid_cols_idx = np.array([1, 2])
        self.offset = 5.0

    def test_init_valid(self):
        qubo = QUBO(self.valid_mat, self.valid_rows_idx, self.valid_cols_idx, self.offset)
        self.assertEqual(qubo.problem_size, 2)
        self.assertEqual(qubo.offset, 5.0)
        self.assertIsNone(qubo.solutions)
        np.testing.assert_array_equal(qubo.rows_idx, self.valid_rows_idx)
        np.testing.assert_array_equal(qubo.cols_idx, self.valid_cols_idx)
        np.testing.assert_array_almost_equal(qubo.mat, self.valid_mat)

    def test_init_invalid_non_square_mat(self):
        non_square_mat = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaisesRegex(AssertionError, "Problem matrix must be square"):
            QUBO(non_square_mat, np.array([1, 2]), np.array([1, 2, 3]))

    def test_init_invalid_col_length(self):
        invalid_cols_idx = np.array([1, 2, 3])
        with self.assertRaisesRegex(AssertionError, "Invalid numer of columns, must be as long as cols_idx array"):
            QUBO(self.valid_mat, self.valid_rows_idx, invalid_cols_idx)

    def test_init_invalid_row_length(self):
        invalid_rows_idx = np.array([1, 2, 3])
        with self.assertRaisesRegex(AssertionError, "Invalid numer of rows, must be as long as rows_idx array"):
            QUBO(self.valid_mat, invalid_rows_idx, self.valid_cols_idx)

    def test_sanitize_already_upper_triangular(self):
        mat = np.array([[1.0, 2.0], [0.0, 3.0]])
        rows = np.array([1, 2])
        cols = np.array([1, 2])
        new_mat, new_cols, new_rows = QUBO.sanitize(mat, cols, rows)
        np.testing.assert_array_almost_equal(new_mat, mat)
        np.testing.assert_array_equal(new_rows, rows)
        np.testing.assert_array_equal(new_cols, cols)

    def test_sanitize_requires_upper_triangular(self):
        mat = np.array([[1.0, 2.0], [4.0, 3.0]])
        rows = np.array([1, 2])
        cols = np.array([1, 2])
        new_mat, new_cols, new_rows = QUBO.sanitize(mat, cols, rows)
        self.assertTrue(np.allclose(new_mat, np.triu(new_mat)))
        self.assertEqual(new_mat.shape, mat.shape)

    def test_sanitize_odd_size_padding(self):
        mat = np.array([[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]])
        rows = np.array([1, 2, 3])
        cols = np.array([1, 2, 3])
        new_mat, new_cols, new_rows = QUBO.sanitize(mat, cols, rows)
        self.assertEqual(new_mat.shape, (4, 4))
        self.assertEqual(new_rows.shape, (4,))
        self.assertEqual(new_cols.shape, (4,))
        np.testing.assert_array_equal(new_rows[-1], -1)
        np.testing.assert_array_equal(new_cols[-1], -1)
        expected_mat = np.zeros((4, 4))
        expected_mat[:3, :3] = mat
        np.testing.assert_array_almost_equal(new_mat, expected_mat)

    def test_sanitize_odd_size_after_triangularization(self):
        mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        rows = np.array([1, 2, 3])
        cols = np.array([1, 2, 3])
        new_mat, new_cols, new_rows = QUBO.sanitize(mat, cols, rows)
        diag = np.diag(np.diag(mat))
        triangular_mat = np.triu(mat + mat.T) - diag
        expected_mat = np.zeros((4, 4))
        expected_mat[:3, :3] = triangular_mat
        self.assertEqual(new_mat.shape, (4, 4))
        np.testing.assert_array_almost_equal(new_mat, expected_mat)
        np.testing.assert_array_equal(new_rows[-1], -1)

    def test_str_representation(self):
        qubo = QUBO(self.valid_mat, self.valid_rows_idx, self.valid_cols_idx, self.offset)
        expected_str = "QUBO(cols: [1 2], rows: [1 2], offset: 5.0, size: 2)"
        self.assertEqual(str(qubo), expected_str)


if __name__ == "__main__":
    unittest.main()
