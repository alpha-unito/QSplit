import unittest

import numpy as np
import pandas as pd

from qsplit.qubo import QUBO
from qsplit.resolve_conflicts import local_search


class TestConflictResolution(unittest.TestCase):
    def setUp(self):
        self.mat_3x3 = np.array([[1.0, 0.5, 0.0], [0.0, 2.0, 1.0], [0.0, 0.0, 3.0]])
        self.qubo_indices = np.array([1, 2, 3])
        self.qubo_obj = QUBO(self.mat_3x3, self.qubo_indices, self.qubo_indices)

    def test_no_nan_energy_recalculation(self):
        initial_df = pd.DataFrame({1: [1, 0], 2: [0, 1], 3: [1, 1], 'energy': [99.0, 99.0]})
        expected_e0 = 4.0
        expected_e1 = 6.0

        expected_df = initial_df.copy()
        expected_df.loc[0, 'energy'] = expected_e0
        expected_df.loc[1, 'energy'] = expected_e1

        result_df = local_search(initial_df.copy(), self.qubo_obj)

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_nan_conflict_resolution(self):
        initial_df = pd.DataFrame({1: [1.0, 0.0], 2: [np.nan, 1], 3: [1, np.nan], 'energy': [99.0, 99.0]})
        result_df = local_search(initial_df.copy(), self.qubo_obj)
        expected_df = pd.DataFrame({1: [1.0, 0.0], 2: [0.0, 1.0], 3: [1.0, 0.0], 'energy': [4.0, 2.0]})
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_mixed_nan_and_no_nan_rows(self):
        initial_df = pd.DataFrame(
            {1: [1, 0, np.nan], 2: [np.nan, 1, np.nan], 3: [1.0, 1.0, 1.0], 'energy': [99.0, 99.0, 99.0]})
        result_df = local_search(initial_df.copy(), self.qubo_obj)
        expected_df = pd.DataFrame(
            {1: [1.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0], 3: [1.0, 1.0, 1.0], 'energy': [4.0, 6.0, 3.0]})

        pd.testing.assert_frame_equal(result_df, expected_df)
