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

from qsplit.conflict_resolution.local_search import nan_subqubo
from qsplit.qubo import QUBO


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

        result_df = nan_subqubo(initial_df.copy(), self.qubo_obj)

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_nan_conflict_resolution(self):
        initial_df = pd.DataFrame({1: [1.0, 0.0], 2: [np.nan, 1], 3: [1, np.nan], 'energy': [99.0, 99.0]})
        result_df = nan_subqubo(initial_df.copy(), self.qubo_obj)
        expected_df = pd.DataFrame({1: [1.0, 0.0], 2: [0.0, 1.0], 3: [1.0, 0.0], 'energy': [4.0, 2.0]})
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_mixed_nan_and_no_nan_rows(self):
        initial_df = pd.DataFrame(
            {1: [1, 0, np.nan], 2: [np.nan, 1, np.nan], 3: [1.0, 1.0, 1.0], 'energy': [99.0, 99.0, 99.0]})
        result_df = nan_subqubo(initial_df.copy(), self.qubo_obj)
        expected_df = pd.DataFrame(
            {1: [1.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0], 3: [1.0, 1.0, 1.0], 'energy': [4.0, 6.0, 3.0]})

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_different_rows_cols(self):
        df = pd.DataFrame.from_dict({'0': [np.nan], '1': [np.nan], '6': [np.nan], '7': [0.0], 'energy': [np.nan]})
        problem = QUBO(mat=np.array(
            [[-68.79627191, -68.80109593, -88.38327757, 73.23522915], [0, -124.02536823, -149.05350201, 7.45948431],
             [0, 0, 128.9931199, -52.7871462], [0, 0, 0, 9.09390549]]), cols_idx=np.array([4, 5, 6, 7]),
            rows_idx=np.array([0, 1, 2, 3]))
        result_df = nan_subqubo(df, problem)
        print(result_df)
        expected_df = pd.DataFrame.from_dict({'0': [0.0], '1': [0.0], '6': [0.0], '7': [0.0], 'energy': [0.0]})
        print(expected_df)
        pd.testing.assert_frame_equal(result_df, expected_df)
