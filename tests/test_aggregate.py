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

from qsplit.aggregation.aggregate_recursive import (__fill_with_nan as fill_with_nan, __combine_ul_lr as combine_ul_lr,
                                                    __nan_hamming_distance as nan_hamming_distance,
                                                    __get_closest_assignments as get_closest_assignments,
                                                    __combine_rows as combine_rows, aggregate_solutions)
from qsplit.qubo import QUBO
from qsplit.splitting.split_recursive import split_problem


class TestAggregateSolutions(unittest.TestCase):
    def test_aggregate(self):
        original_problem = np.array([[-302, 132, 168, 228], [0, -949, 616, 836], [0, 0, -1074, 1064], [0, 0, 0, -1259]])
        original_problem_ids = np.array([0, 1, 2, 3])
        original_qubo = QUBO(original_problem, original_problem_ids.copy(), original_problem_ids.copy())

        ul_qubo, ur_qubo, lr_qubo = split_problem(original_qubo)

        ul_qubo.solutions = pd.DataFrame({0: [1], 1: [1], 'energy': [-1119]})
        ur_qubo.solutions = pd.DataFrame(
            {0: [0, 0, 0, 0, 0, 0, 0, 0, 1, 1], 1: [0, 0, 0, 0, 1, 1, 1, 1, 0, 0], 2: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
             3: [0, 1, 0, 1, 0, 1, 0, 1, 0, 0], 'energy': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]})
        lr_qubo.solutions = pd.DataFrame({3: [1], 4: [1], 'energy': [-1269.0]})

        result_qubo = aggregate_solutions((ul_qubo, ur_qubo, lr_qubo), original_qubo)

        expected_qubo = QUBO(original_problem, original_problem_ids, original_problem_ids)
        expected_qubo.solutions = pd.DataFrame({0: [1.0], 1: [1.0], 2: [0], 3: [1.0], 'energy': [-1314.0]})
        pd.testing.assert_frame_equal(result_qubo.solutions, expected_qubo.solutions)


class TestCombineRows(unittest.TestCase):
    def setUp(self):
        self.base_index = ['A', 'B', 'C', 'energy']

    def test_variables_perfect_match(self):
        row1 = pd.Series([1, 0, 1, 10.0], index=self.base_index)
        row2 = pd.Series([1, 0, 1, 5.0], index=self.base_index)
        result = combine_rows(row1, row2)
        expected = [1, 0, 1, 15.0]
        np.testing.assert_array_equal(result, expected)

    def test_variables_conflict(self):
        row1 = pd.Series([1, 0, 1, 10.0], index=self.base_index)
        row2 = pd.Series([0, 0, 1, 5.0], index=self.base_index)
        result = combine_rows(row1, row2)
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[3]))
        self.assertEqual(result[1], 0)
        self.assertEqual(result[2], 1)

    def test_variables_one_is_nan(self):
        row1 = pd.Series([1, np.nan, 1, 10.0], index=self.base_index)
        row2 = pd.Series([1, 0, np.nan, 5.0], index=self.base_index)
        result = combine_rows(row1, row2)
        expected = [1, 0, 1, 15.0]
        np.testing.assert_array_equal(result, expected)

    def test_energy_sum_valid(self):
        row1 = pd.Series([1, 1, 1, 1.5], index=self.base_index)
        row2 = pd.Series([1, 1, 1, 2.5], index=self.base_index)
        result = combine_rows(row1, row2)
        self.assertAlmostEqual(result[-1], 4.0)

    def test_energy_one_is_nan(self):
        row1 = pd.Series([1, 1, 1, np.nan], index=self.base_index)
        row2 = pd.Series([1, 1, 1, 5.0], index=self.base_index)
        result1 = combine_rows(row1, row2)
        result2 = combine_rows(row2, row1)
        self.assertEqual(result1[-1], 5.0)
        self.assertEqual(result2[-1], 5.0)

    def test_energy_both_are_nan(self):
        row1 = pd.Series([1, 1, 1, np.nan], index=self.base_index)
        row2 = pd.Series([1, 1, 1, np.nan], index=self.base_index)
        result = combine_rows(row1, row2)
        self.assertTrue(np.isnan(result[-1]))

    def test_energy_nan_due_to_variable_conflict(self):
        row1 = pd.Series([1, 0, 1, 10.0], index=self.base_index)
        row2 = pd.Series([0, 0, 1, 5.0], index=self.base_index)
        result = combine_rows(row1, row2)
        self.assertTrue(np.isnan(result[-1]))

    def test_complex_mix_of_nan_and_conflict(self):
        row1 = pd.Series([1, np.nan, 0, 10.0], index=self.base_index)
        row2 = pd.Series([0, 1, np.nan, np.nan], index=self.base_index)
        result = combine_rows(row1, row2)
        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(result[1], 1)
        self.assertEqual(result[2], 0)
        self.assertTrue(np.isnan(result[3]))

    def test_different_data_types(self):
        row1 = pd.Series([1, 0, 1.0, 10.0], index=self.base_index)
        row2 = pd.Series([1.0, 0.0, 1, 5.0], index=self.base_index)
        result = combine_rows(row1, row2)
        expected = [1.0, 0.0, 1.0, 15.0]
        np.testing.assert_array_equal(result, expected)


class TestGetClosestAssignments(unittest.TestCase):
    def test_perfect_match_no_nan(self):
        start_data = {'A': [0, 1], 'B': [1, 0], 'C': [0, 1], 'energy': [10.0, 20.0]}
        starting_sols = pd.DataFrame(start_data)
        ur_data = {'A': [1, 0, 1], 'B': [0, 1, 1], 'C': [1, 0, 0], 'energy': [15.0, 25.0, 30.0]}
        ur_qubo_filled = pd.DataFrame(ur_data)
        expected_data = {'A': [0.0, 1.0], 'B': [1.0, 0.0], 'C': [0.0, 1.0], 'energy': [25.0, 15.0]}
        expected_df = pd.DataFrame(expected_data)

        result_df = get_closest_assignments(starting_sols, ur_qubo_filled)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_closest_match_with_nan(self):
        start_data = {'A': [1], 'B': [0], 'C': [np.nan], 'D': [1], 'energy': [10.0]}
        starting_sols = pd.DataFrame(start_data)
        ur_data = {'A': [1, 1, 0], 'B': [1, 0, 0], 'C': [0, 1, 1], 'D': [0, 1, 0], 'energy': [5.0, 15.0, 25.0]}
        ur_qubo_filled = pd.DataFrame(ur_data)
        expected_row = ur_qubo_filled.iloc[1].copy()
        expected_df = pd.DataFrame([expected_row])
        expected_df = expected_df.reset_index(drop=True)
        result_df = get_closest_assignments(starting_sols, ur_qubo_filled)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_energy_nan_due_to_unassigned_variable(self):
        start_data = {'A': [0], 'B': [1], 'C': [0], 'energy': [10.0]}
        starting_sols = pd.DataFrame(start_data)
        ur_data = {'A': [1, 0], 'B': [1, np.nan], 'C': [0, 1], 'energy': [5.0, 15.0]}
        ur_qubo_filled = pd.DataFrame(ur_data)
        expected_df = pd.DataFrame({'A': [1.0], 'B': [1.0], 'C': [0.0], 'energy': [5.0]})
        expected_df = expected_df.reset_index(drop=True)
        result_df = get_closest_assignments(starting_sols, ur_qubo_filled)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_all_inf_distances(self):
        start_data = {'A': [np.nan], 'B': [np.nan], 'energy': [10.0]}
        starting_sols = pd.DataFrame(start_data)
        ur_data = {'A': [np.nan, np.nan], 'B': [np.nan, np.nan], 'energy': [5.0, 15.0]}
        ur_qubo_filled = pd.DataFrame(ur_data)
        expected_row = ur_qubo_filled.iloc[0].copy()
        expected_row['energy'] = np.nan
        expected_df = pd.DataFrame([expected_row])
        expected_df = expected_df.reset_index(drop=True)
        result_df = get_closest_assignments(starting_sols, ur_qubo_filled)

        pd.testing.assert_frame_equal(result_df, expected_df)
        self.assertTrue(np.isnan(result_df.loc[0, 'energy']))

    def test_starting_sols_with_multiple_rows(self):
        start_data = {'A': [0, 1, 0], 'B': [0, 1, 1], 'energy': [10.0, 20.0, 30.0]}
        starting_sols = pd.DataFrame(start_data)
        ur_data = {'A': [1, 0], 'B': [1, 0], 'energy': [5.0, 15.0]}
        ur_qubo_filled = pd.DataFrame(ur_data)
        expected_rows = [ur_qubo_filled.iloc[1].copy(), ur_qubo_filled.iloc[0].copy(), ur_qubo_filled.iloc[0].copy()]
        expected_df = pd.DataFrame(expected_rows).reset_index(drop=True)
        result_df = get_closest_assignments(starting_sols, ur_qubo_filled)

        pd.testing.assert_frame_equal(result_df, expected_df)


class TestNanHammingDistance(unittest.TestCase):
    def test_identical_arrays(self):
        a = np.array([1, 0, 1, 0, 1])
        b = np.array([1, 0, 1, 0, 1])
        expected_distance = 0.0
        result = nan_hamming_distance(a, b)
        self.assertAlmostEqual(result, expected_distance)
        self.assertEqual(result, 0.0)

    def test_completely_different_arrays(self):
        a = np.array([1, 0, 1, 0])
        b = np.array([0, 1, 0, 1])
        expected_distance = 1.0
        result = nan_hamming_distance(a, b)
        self.assertAlmostEqual(result, expected_distance)
        self.assertEqual(result, 1.0)

    def test_nan_ignored_in_one_array(self):
        a = np.array([1, 0, 1, 0, 1])
        b = np.array([1, 0, np.nan, 1, 1])
        expected_distance = 1 / 4
        result = nan_hamming_distance(a, b)
        self.assertAlmostEqual(result, expected_distance)

    def test_partial_match_with_nan(self):
        a = np.array([0, 1, 1, np.nan, 0, 1, 5.0])
        b = np.array([0, 0, 1, 1.0, np.nan, 1, 6.0])
        expected_distance = 2 / 5
        result = nan_hamming_distance(a, b)
        self.assertAlmostEqual(result, expected_distance)

    def test_only_nan_overlap(self):
        a = np.array([np.nan, np.nan, 1])
        b = np.array([1, np.nan, np.nan])
        result = nan_hamming_distance(a, b)
        self.assertEqual(result, np.inf)

    def test_empty_arrays(self):
        a = np.array([])
        b = np.array([])
        result = nan_hamming_distance(a, b)
        self.assertEqual(result, np.inf)

    def test_different_data_types(self):
        a = np.array([1, 0, 1.0])
        b = np.array([1.0, 0.0, 0])
        expected_distance = 1 / 3
        result = nan_hamming_distance(a, b)
        self.assertAlmostEqual(result, expected_distance)


class TestCombineUlLr(unittest.TestCase):
    def setUp(self):
        mat = np.eye(2)
        rows_idx = np.array([1, 2])
        cols_idx = np.array([1, 2])
        self.qubo_ul = QUBO(mat, rows_idx, cols_idx)
        self.qubo_lr = QUBO(mat, rows_idx, cols_idx)

    def test_basic_combination(self):
        self.qubo_ul.rows_idx = np.array([1, 2])
        self.qubo_ul.solutions = pd.DataFrame({1: [0, 1], 2: [1, 0], 'energy': [10.0, 20.0]})
        self.qubo_lr.cols_idx = np.array([3, 4])
        self.qubo_lr.solutions = pd.DataFrame({3: [0, 1], 4: [0, 1], 'energy': [5.0, 15.0]})
        combined_df = combine_ul_lr(self.qubo_ul, self.qubo_lr)

        expected_data = {1: [0, 0, 1, 1], 2: [1, 1, 0, 0], 3: [0, 1, 0, 1], 4: [0, 1, 0, 1],
                         'energy': [10.0 + 5.0, 10.0 + 15.0, 20.0 + 5.0, 20.0 + 15.0, ]}
        expected_df = pd.DataFrame(expected_data)
        expected_df = expected_df[[1, 2, 3, 4, 'energy']]

        pd.testing.assert_frame_equal(combined_df, expected_df)
        self.assertNotIn('tmp', self.qubo_ul.solutions.columns)
        self.assertNotIn('tmp', self.qubo_lr.solutions.columns)

    def test_missing_indices(self):
        self.qubo_ul.rows_idx = np.array([1, 4])
        self.qubo_ul.solutions = pd.DataFrame({1: [0], 'energy': [10.0]})
        self.qubo_lr.cols_idx = np.array([2, 5])
        self.qubo_lr.solutions = pd.DataFrame({5: [1], 'energy': [5.0]})
        combined_df = combine_ul_lr(self.qubo_ul, self.qubo_lr)
        expected_data = {1: [0], 2: [np.nan], 4: [np.nan], 5: [1], 'energy': [15.0]}
        expected_df = pd.DataFrame(expected_data)

        pd.testing.assert_frame_equal(combined_df, expected_df)
        self.assertTrue(combined_df[2].isna().all())
        self.assertTrue(combined_df[4].isna().all())
        self.assertEqual(list(combined_df.columns), [1, 2, 4, 5, 'energy'])

    def test_empty_solutions(self):
        self.qubo_ul.rows_idx = np.array([1, 2])
        self.qubo_ul.solutions = pd.DataFrame(columns=[1, 2, 'energy'])
        self.qubo_lr.cols_idx = np.array([3, 4])
        self.qubo_lr.solutions = pd.DataFrame({3: [0], 4: [1], 'energy': [5.0]})
        combined_df = combine_ul_lr(self.qubo_ul, self.qubo_lr)

        expected_cols = [1, 2, 3, 4, 'energy']
        expected_df = pd.DataFrame(columns=expected_cols)

        pd.testing.assert_frame_equal(combined_df, expected_df, check_dtype=False)
        self.assertTrue(combined_df.empty)

    def test_large_combination(self):
        self.qubo_ul.rows_idx = np.array([10])
        self.qubo_ul.solutions = pd.DataFrame({10: [0, 1, 0], 'energy': [1.0, 2.0, 3.0]})
        self.qubo_lr.cols_idx = np.array([20])
        self.qubo_lr.solutions = pd.DataFrame({20: [0, 1], 'energy': [10.0, 20.0]})
        combined_df = combine_ul_lr(self.qubo_ul, self.qubo_lr)
        expected_data = {10: [0, 0, 1, 1, 0, 0], 20: [0, 1, 0, 1, 0, 1],
                         'energy': [1.0 + 10.0, 1.0 + 20.0, 2.0 + 10.0, 2.0 + 20.0, 3.0 + 10.0, 3.0 + 20.0]}
        expected_df = pd.DataFrame(expected_data)[[10, 20, 'energy']]
        pd.testing.assert_frame_equal(combined_df, expected_df)


class TestFillWithNan(unittest.TestCase):
    def test_no_missing_columns(self):
        df = pd.DataFrame({'B': [1, 2], 'A': [3, 4], 'C': [5, 6]})
        schema = pd.Index(['A', 'B', 'C'])
        expected_df = pd.DataFrame({'A': [3, 4], 'B': [1, 2], 'C': [5, 6]})
        result_df = fill_with_nan(schema, df)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_missing_columns_are_added(self):
        df = pd.DataFrame({'ID': [101, 102], 'Val_A': [5.0, 10.0]})
        schema = pd.Index(['ID', 'Val_B', 'Val_A', 'Val_C'])
        expected_df = pd.DataFrame(
            {'ID': [101, 102], 'Val_B': [np.nan, np.nan], 'Val_A': [5.0, 10.0], 'Val_C': [np.nan, np.nan]})
        result_df = fill_with_nan(schema, df)
        pd.testing.assert_frame_equal(result_df, expected_df)
        self.assertTrue(result_df['Val_B'].isna().all())
        self.assertTrue(result_df['Val_C'].isna().all())

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=['A', 'B'])
        schema = pd.Index(['A', 'B', 'C', 'D'])
        result_df = fill_with_nan(schema, df)
        expected_df = pd.DataFrame(columns=['A', 'B', 'C', 'D'])
        pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=False)
        self.assertEqual(list(result_df.columns), list(schema))

    def test_extra_columns_in_dataframe(self):
        df = pd.DataFrame({'A': [1], 'B': [2], 'D': [4], 'E': [5]})
        schema = pd.Index(['A', 'B', 'C', 'D', 'E'])
        result_df = fill_with_nan(schema, df)
        expected_df = pd.DataFrame({'A': [1], 'B': [2], 'C': [np.nan], 'D': [4], 'E': [5]})
        pd.testing.assert_frame_equal(result_df, expected_df)
        self.assertTrue(result_df['C'].isna().all())

    def test_schema_single_column(self):
        df = pd.DataFrame({'A': [10, 20], 'B': [30, 40]})
        schema = pd.Index(['B'])
        result_df = fill_with_nan(schema, df)
        expected_df = pd.DataFrame({'B': [30, 40]})
        pd.testing.assert_frame_equal(result_df, expected_df)


if __name__ == '__main__':
    unittest.main()
