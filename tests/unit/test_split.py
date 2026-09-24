import os
import unittest
from unittest.mock import patch

import numpy as np

from qsplit.qubo import QUBO
from qsplit.splitting.split_linear import split_problem as split_linear
from qsplit.splitting.split_recursive import split_problem


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
        self.assertEqual(np.count_nonzero(res_ul.mat), 1)
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
        self.assertEqual(np.count_nonzero(res_ul.mat), 1)
        self.assertEqual(res_ul.cols_idx.tolist(), [0, 1, 2])

        # UR check
        self.assertEqual(res_ur.mat[0, 0], 2.0)
        self.assertEqual(np.count_nonzero(res_ur.mat), 1)
        self.assertEqual(res_ur.cols_idx.tolist(), [3, 4, -1])
        self.assertEqual(res_ur.rows_idx.tolist(), [0, 1, 2])

        # LR check
        self.assertEqual(res_lr.mat[0, 0], 3.0)
        self.assertEqual(np.count_nonzero(res_lr.mat), 1)
        self.assertEqual(res_lr.rows_idx.tolist(), [3, 4, -1])
        self.assertEqual(qubo.problem_size, 5)
        np.testing.assert_array_equal(qubo.mat, mat)
        np.testing.assert_array_equal(qubo.rows_idx, np.arange(5))
        np.testing.assert_array_equal(qubo.cols_idx, np.arange(5))

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

        self.assertEqual(ul.cols_idx.tolist(), [0, 1, 2])
        self.assertEqual(ul.rows_idx.tolist(), [0, 1, 2])

        self.assertEqual(ur.cols_idx.tolist(), [3, 4, 5])
        self.assertEqual(ur.rows_idx.tolist(), [0, 1, 2])

        self.assertEqual(lr.cols_idx.tolist(), [3, 4, 5])
        self.assertEqual(lr.rows_idx.tolist(), [3, 4, 5])

    def test_recursive_solver_skips_leaf_with_only_padding_biases(self):
        from qsplit.local_runner import qsplit_sampler_recursive

        # The lower half of the UR block is folded into its padding column.
        mat = np.zeros((5, 5))
        mat[2, 3] = 2.0
        _, ur, _ = split_problem(QUBO(mat, np.arange(5), np.arange(5)))
        self.assertTrue(np.any(ur.mat))
        with patch.dict(os.environ, {"CUT_DIM": "10"}), patch("qsplit.local_runner.solve") as sampler:
            result = qsplit_sampler_recursive(ur)
        sampler.assert_not_called()
        self.assertTrue(result.solutions.isna().all().all())


class TestSplitLinear(unittest.TestCase):
    def test_split_simple_blocks(self):
        mat = np.zeros((4, 4))
        mat[0, 0] = 10.0
        mat[2, 2] = 20.0
        ids = np.array([10, 11, 12, 13])
        qubo = QUBO(mat, ids.copy(), ids.copy())
        os.environ["CUT_DIM"] = "2"
        res = split_linear(qubo)

        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].mat[0, 0], 10.0)
        self.assertEqual(res[1].mat[0, 0], 20.0)
        self.assertEqual(res[0].rows_idx.tolist(), [10, 11])
        self.assertEqual(res[1].rows_idx.tolist(), [12, 13])

    def test_padding_logic(self):
        mat = np.eye(3)
        ids = np.array([1, 2, 3])
        qubo = QUBO(mat, ids.copy(), ids.copy())
        os.environ["CUT_DIM"] = "2"
        res = split_linear(qubo)

        self.assertEqual(len(res), 2)
        self.assertEqual(res[1].rows_idx.tolist(), [3, -1])
        self.assertEqual(res[1].mat.shape, (2, 2))

    def test_skip_empty_blocks(self):
        mat = np.zeros((4, 4))
        mat[0, 0] = 5.0
        qubo = QUBO(mat, np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))
        os.environ["CUT_DIM"] = "2"
        res = split_linear(qubo)

        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].rows_idx.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
