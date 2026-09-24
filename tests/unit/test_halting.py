import os
import unittest

import numpy as np

from qsplit.halting_heuristic.stop import is_empty, is_sparse, vars_count
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

    def test_padding_only_biases_are_empty(self):
        for rows, cols in [([0, 1], [2, -1]), ([-1, 0], [1, 2]), ([-1, -2], [-1, -2])]:
            with self.subTest(rows=rows, cols=cols):
                q = QUBO(np.array([[0.0, 3.0], [0.0, 0.0]]), np.array(rows), np.array(cols))
                self.assertTrue(is_empty(q))
                self.assertEqual(vars_count(q), 0)

    def test_padding_masks_are_independent(self):
        q = QUBO(np.array([[0.0, 3.0], [0.0, 0.0]]), np.array([0, -1]), np.array([-1, 1]))
        self.assertFalse(is_empty(q))
        self.assertEqual(vars_count(q), 2)

    def test_zero_dimension_is_empty(self):
        q = QUBO(np.zeros((0, 0)), np.array([], dtype=int), np.array([], dtype=int))
        self.assertTrue(is_empty(q))
        self.assertEqual(vars_count(q), 0)

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
