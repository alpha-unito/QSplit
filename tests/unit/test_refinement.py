import os
import unittest
from copy import deepcopy
from itertools import product
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from dimod import ExactSolver

from qsplit import local_runner
from qsplit.adapters.dwave.util import from_qubo_matrix_to_bqm, to_dataframe
from qsplit.qubo import QUBO
from qsplit.refinement.propagation import collect_beliefs, condition_subproblem
from qsplit.refinement.refine_conditioned import refine_solutions
from qsplit.refinement.refine_linear import refine_problems as refine_linear
from qsplit.refinement.refine_quadtree import refine_problems as refine_quadtree
from qsplit.splitting.split_quadtree import split_problem as split_quadtree


def exact_solve(qubo):
    return to_dataframe(ExactSolver().sample(from_qubo_matrix_to_bqm(qubo)), qubo)


def make_qubo(matrix, ids=None, offset=0.0):
    matrix = np.array(matrix, dtype=float)
    ids = np.arange(len(matrix)) if ids is None else np.array(ids)
    return QUBO(matrix, ids.copy(), ids.copy(), offset=offset)


class TestPropagation(unittest.TestCase):
    def test_conditioned_energy_equals_global_energy_for_every_local_assignment(self):
        qubo = make_qubo([[2, -4, 3, 0], [0, 1, -2, 0], [0, 0, -1, 0], [0, 0, 0, 0]], ids=[20, 10, 30, -1], offset=7)
        fixed = {20: 0, 10: 1, 30: 0}
        sub = condition_subproblem(qubo, [30, 20], fixed)
        for bits in product((0, 1), repeat=2):
            full = {**fixed, **dict(zip(sub.rows_idx, bits))}
            x = np.array([full.get(idx, 0) for idx in qubo.rows_idx])
            local = np.array(bits)
            self.assertEqual(local @ sub.mat @ local + sub.offset, x @ qubo.mat @ x + qubo.offset)

    def test_sequential_updates_do_not_combine_incompatible_local_improvements(self):
        qubo = make_qubo([[-1, 3], [0, -1]])
        qubo.solutions = pd.DataFrame({0: [0], 1: [0], "energy": [0.0]})
        result = refine_solutions(qubo, exact_solve, 1, np.random.default_rng(0))
        # Either isolated flip improves energy, but taking both would give +1.
        self.assertEqual(result.iloc[0]["energy"], -1)
        self.assertEqual(result.iloc[0][0] + result.iloc[0][1], 1)
        self.assertEqual(qubo.solutions.iloc[0]["energy"], 0)

    def test_joint_updates_cross_a_single_variable_barrier(self):
        qubo = make_qubo([[1, -3], [0, 1]])
        qubo.solutions = pd.DataFrame({0: [0], 1: [0], "energy": [0.0]})
        result = refine_solutions(qubo, exact_solve, 2, np.random.default_rng(0))
        self.assertEqual(result.iloc[0].tolist(), [1, 1, -1])

    def test_conditioned_refinement_rejects_a_worsening_sampler_result(self):
        qubo = make_qubo([[-1, 0], [0, -2]])
        qubo.solutions = pd.DataFrame({0: [1], 1: [1], "energy": [-3.0]})
        solve = Mock(return_value=pd.DataFrame({0: [0], 1: [0], "energy": [-999.0]}))
        result = refine_solutions(qubo, solve, 2, np.random.default_rng(0))
        pd.testing.assert_frame_equal(result, qubo.solutions, check_dtype=False)

    def test_beliefs_weight_each_subproblem_once_and_ignore_invalid_samples(self):
        qubo = make_qubo(np.eye(2))
        qubo.solutions = pd.DataFrame({0: [1], 1: [0], "energy": [1.0]})
        first = make_qubo(np.eye(2))
        first.solutions = pd.DataFrame({0: [0, 0, 1], 1: [np.nan, 2, 1], "energy": [0, 0, np.inf]})
        second = make_qubo([[1]], ids=[0])
        second.solutions = pd.DataFrame({0: [1], "energy": [100.0]})
        missing = make_qubo([[1]])
        missing.solutions = pd.DataFrame({0: [np.nan], "energy": [np.nan]})
        self.assertEqual(collect_beliefs([first, second, missing], qubo), {0: 0.75, 1: 0.0})

    def test_linear_restores_cut_couplings_and_uses_shared_beliefs(self):
        qubo = make_qubo([[2, -4], [0, 3]], ids=[20, 10], offset=7)
        qubo.solutions = pd.DataFrame({20: [0], 10: [1], "energy": [10.0]})
        first = QUBO(np.array([[-4.0]]), np.array([20]), np.array([10]))
        first.solutions = pd.DataFrame({10: [0], "energy": [0.0]})
        second = make_qubo([[3]], ids=[10])
        second.solutions = pd.DataFrame({10: [1], "energy": [3.0]})
        before = qubo.mat.copy()

        refined = refine_linear([first, second], qubo)
        self.assertEqual(len(refined), 2)
        # p_10 = (global 1 + local mean 0.5) / 2 = 0.75.
        self.assertEqual(refined[0].mat[0, 0], 2 - 4 * 0.75)
        self.assertEqual(refined[0].offset, 7 + 3 * 0.75)
        self.assertEqual(refined[0].rows_idx.tolist(), [20])
        self.assertEqual(refined[1].rows_idx.tolist(), [10])
        np.testing.assert_array_equal(qubo.mat, before)
        np.testing.assert_array_equal(first.mat, [[-4]])
        self.assertTrue(all(sub.solutions is None for sub in refined))

    def test_linear_skips_padding_and_rebuilds_internal_interactions(self):
        qubo = make_qubo([[1, 4, 0], [0, 2, -3], [0, 0, 5]])
        qubo.solutions = pd.DataFrame({0: [1], 1: [0], 2: [0], "energy": [1.0]})
        sub = QUBO(np.eye(2), np.array([0, 1]), np.array([2, -1]))
        refined = refine_linear([sub], qubo)
        np.testing.assert_array_equal(refined[0].mat, [[1, 4], [0, 2]])
        self.assertEqual(refined[1].rows_idx.tolist(), [2])
        self.assertTrue(all(sub.problem_size <= 2 for sub in refined))

    def test_quadtree_macro_members_are_local_and_penalties_do_not_accumulate(self):
        qubo = make_qubo([[2, -4, 1, 3], [0, 1, -2, 4], [0, 0, -1, 2], [0, 0, 0, 1]], offset=5)
        qubo.solutions = pd.DataFrame({0: [0], 1: [1], 2: [0], 3: [1], "energy": [11.0]})
        with patch.dict(os.environ, {"CUT_DIM": "2", "EXACT_RATIO": "0.5", "REFINEMENT_STRENGTH": "0"}):
            subs = split_quadtree(qubo)
            self.assertEqual(subs[0].macro_members[-1000], [1, 3, 2])
            self.assertNotEqual(subs[0].macro_members[-1000], subs[1].macro_members[-1000])
            rebuilt = refine_quadtree(subs, qubo)
            for original, new in zip(subs, rebuilt):
                np.testing.assert_allclose(original.mat, new.mat)
                self.assertEqual(new.offset, qubo.offset)

        with patch.dict(os.environ, {"REFINEMENT_STRENGTH": "0.2"}):
            first = refine_quadtree(subs, qubo)
            second = refine_quadtree(first, qubo)
        for original, new, repeated in zip(subs, first, second):
            targets = np.array(
                [
                    qubo.solutions.iloc[0][original.rows_idx[0]],
                    np.mean([qubo.solutions.iloc[0][idx] for idx in original.macro_members[-1000]]),
                ]
            )
            magnitudes = np.abs(original.mat)
            scale = magnitudes.sum(axis=0) + magnitudes.sum(axis=1) - np.diag(magnitudes)
            np.testing.assert_allclose(np.diag(new.mat - original.mat), 0.2 * scale * (1 - 2 * targets))
            np.testing.assert_allclose(new.mat, repeated.mat)
            self.assertEqual(new.offset, repeated.offset)
            self.assertEqual(new.macro_members, original.macro_members)


class TestRefinedSamplers(unittest.TestCase):
    samplers = (local_runner.qsplit_sampler_refined_iterative, local_runner.qsplit_sampler_refined_quadtree)

    def test_disabled_refinement_delegates_without_reading_other_settings(self):
        for refined, legacy in zip(self.samplers, ("qsplit_sampler_iterative", "qsplit_sampler_quadtree")):
            for loops in (None, "0", "-2"):
                with self.subTest(sampler=legacy, loops=loops):
                    env = {
                        "REFINEMENT_TOLERANCE": "invalid",
                        "REFINEMENT_STRENGTH": "invalid",
                        "REFINEMENT_METHOD": "invalid",
                        "REFINEMENT_PATIENCE": "invalid",
                    }
                    if loops is not None:
                        env["REFINEMENT_LOOPS"] = loops
                    strategy = Mock(side_effect=AssertionError("Refinement must be disabled"))
                    with patch.dict(os.environ, env, clear=True), patch.object(local_runner, legacy) as previous:
                        qubo = make_qubo([[1]])
                        self.assertIs(refined(qubo, refinement=strategy), previous.return_value)
                        previous.assert_called_once_with(qubo)
                        strategy.assert_not_called()

    def test_loop_limit_convergence_and_best_solution_retention(self):
        cases = [
            ("2", "0", [1, 2, 3], [-1, -2, -3], -3),
            ("8", "0", [1, 2, 1], [-1, -2, -1], -2),
            ("8", "0", [1, 1], [-1, -1], -1),
            ("8", "1.1", [1, 2], [-1, -2], -2),
        ]
        for sampler in self.samplers:
            for loops, tolerance, counts, expected_history, expected in cases:
                with self.subTest(sampler=sampler.__name__, loops=loops, counts=counts):
                    samples = [
                        pd.DataFrame({**{idx: [int(idx < count)] for idx in range(4)}, "energy": [-999.0]})
                        for count in counts
                    ]
                    strategy = Mock(side_effect=lambda subs, qubo: deepcopy(subs))
                    env = {"CUT_DIM": "4", "REFINEMENT_LOOPS": loops, "REFINEMENT_TOLERANCE": tolerance}
                    with (
                        patch.dict(os.environ, env, clear=True),
                        patch.object(local_runner, "solve", side_effect=samples) as solve,
                    ):
                        result = sampler(make_qubo(-np.eye(4), offset=7), refinement=strategy)
                    self.assertEqual(result.refinement_history, [energy + 7 for energy in expected_history])
                    self.assertEqual(result.solutions.iloc[0]["energy"], expected + 7)
                    self.assertEqual(solve.call_count, len(expected_history))
                    self.assertEqual(strategy.call_count, len(expected_history) - 1)

    def test_default_strategies_improve_known_instances(self):
        matrices = [
            [
                [-1, -4, -1, 5, -4, -4],
                [0, -3, -5, -1, -4, 1],
                [0, 0, 1, -5, -5, 5],
                [0, 0, 0, -2, 0, -3],
                [0, 0, 0, 0, 4, -3],
                [0, 0, 0, 0, 0, 2],
            ],
            [
                [-2, 1, -5, 3, 1, 1],
                [0, 1, -5, 3, 1, 1],
                [0, 0, 4, 1, -1, 1],
                [0, 0, 0, 2, -2, -1],
                [0, 0, 0, 0, 1, 3],
                [0, 0, 0, 0, 0, 3],
            ],
        ]
        for sampler, matrix, expected in zip(self.samplers, matrices, (-27, -6)):
            with self.subTest(sampler=sampler.__name__):
                env = {
                    "CUT_DIM": "4",
                    "REFINEMENT_LOOPS": "4",
                    "REFINEMENT_STRENGTH": "0.1",
                    "REFINEMENT_METHOD": "consensus",
                }
                with (
                    patch.dict(os.environ, env, clear=True),
                    patch.object(local_runner, "solve", side_effect=exact_solve),
                ):
                    result = sampler(make_qubo(matrix, ids=[30, 10, 50, 20, 60, 40], offset=3))
                self.assertLess(min(result.refinement_history), result.refinement_history[0])
                self.assertEqual(result.solutions.iloc[0]["energy"], expected + 3)
                assignment = result.solutions.loc[0, [30, 10, 50, 20, 60, 40]].to_numpy()
                self.assertEqual(assignment @ np.array(matrix) @ assignment + 3, expected + 3)
                self.assertTrue(all(idx >= 0 for idx in result.solutions.columns if idx != "energy"))

    def test_zero_objective_and_cancelled_local_fields(self):
        env = {"CUT_DIM": "2", "REFINEMENT_LOOPS": "3"}
        for sampler in self.samplers:
            with self.subTest(sampler=sampler.__name__), patch.dict(os.environ, env, clear=True):
                with patch.object(local_runner, "solve") as solve:
                    result = sampler(make_qubo(np.zeros((3, 3)), offset=4))
                solve.assert_not_called()
                self.assertEqual(result.solutions.iloc[0]["energy"], 4)
        with (
            patch.dict(os.environ, {**env, "CUT_DIM": "1", "REFINEMENT_METHOD": "consensus"}, clear=True),
            patch.object(local_runner, "solve", side_effect=exact_solve) as solve,
        ):
            result = local_runner.qsplit_sampler_refined_iterative(make_qubo([[2, -2], [0, -1]]))
        self.assertEqual(solve.call_count, 4)
        self.assertTrue(np.isfinite(result.solutions.to_numpy()).all())

    def test_belief_propagation_aggregation_is_preserved(self):
        env = {"CUT_DIM": "2", "REFINEMENT_LOOPS": "1", "REFINEMENT_METHOD": "consensus"}
        with patch.dict(os.environ, env, clear=True), patch.object(local_runner, "BP", True):
            with (
                patch.object(local_runner, "solve", side_effect=exact_solve),
                patch.object(
                    local_runner, "aggregate_solutions_linear_bp", wraps=local_runner.aggregate_solutions_linear_bp
                ) as aggregate,
            ):
                result = local_runner.qsplit_sampler_refined_iterative(make_qubo([[-1, 0], [0, -2]]))
        self.assertEqual(aggregate.call_count, 2)
        self.assertEqual(result.refinement_history, [-3, -3])

    def test_conditioned_refinement_waits_for_multiple_stalled_sweeps(self):
        for patience, expected_calls, expected_energy in ((2, 2, 0), (3, 3, -1)):
            env = {"CUT_DIM": "2", "REFINEMENT_LOOPS": "3", "REFINEMENT_PATIENCE": str(patience)}
            initial = pd.DataFrame({0: [0], "energy": [0.0]})
            improved = pd.DataFrame({0: [1], "energy": [-1.0]})
            with (
                patch.dict(os.environ, env, clear=True),
                patch.object(local_runner, "solve", return_value=initial),
                patch.object(
                    local_runner, "refine_solutions_conditioned", side_effect=[initial, initial, improved]
                ) as refine,
            ):
                result = local_runner.qsplit_sampler_refined_quadtree(make_qubo([[-1]]))
            self.assertEqual(refine.call_count, expected_calls)
            self.assertEqual(result.solutions.iloc[0]["energy"], expected_energy)

    def test_conditioned_neighborhoods_change_and_respect_cut_dim(self):
        qubo = make_qubo(-np.eye(6), ids=[60, 10, 50, 20, 40, 30])
        env = {"CUT_DIM": "2", "REFINEMENT_LOOPS": "3", "REFINEMENT_SEED": "42"}
        for sampler in self.samplers:
            windows = []

            def solve(sub):
                windows.append(tuple(sub.cols_idx))
                self.assertLessEqual(sub.problem_size, 2)
                return exact_solve(sub)

            with patch.dict(os.environ, env, clear=True), patch.object(local_runner, "solve", side_effect=solve):
                result = sampler(deepcopy(qubo))
            self.assertEqual(len(result.refinement_history), 4)
            self.assertTrue(np.all(np.diff(result.refinement_history) <= 0))
            self.assertNotEqual(set(windows[-9:-6]), set(windows[-6:-3]))
            self.assertEqual(result.solutions.iloc[0]["energy"], -6)

    def test_invalid_configuration_fails_explicitly(self):
        for name, value, sampler in (
            ("REFINEMENT_LOOPS", "1.5", self.samplers[0]),
            ("REFINEMENT_TOLERANCE", "nan", self.samplers[0]),
            ("REFINEMENT_TOLERANCE", "-1", self.samplers[1]),
            ("REFINEMENT_STRENGTH", "-1", self.samplers[1]),
            ("REFINEMENT_STRENGTH", "inf", self.samplers[1]),
            ("REFINEMENT_METHOD", "unknown", self.samplers[0]),
            ("REFINEMENT_PATIENCE", "0", self.samplers[0]),
        ):
            with self.subTest(setting=name, value=value):
                env = {"CUT_DIM": "2", "REFINEMENT_LOOPS": "1", "REFINEMENT_METHOD": "consensus", name: value}
                with (
                    patch.dict(os.environ, env, clear=True),
                    patch.object(local_runner, "solve", side_effect=exact_solve),
                    self.assertRaises(ValueError),
                ):
                    sampler(make_qubo([[-1, 0], [0, -2]]))


if __name__ == "__main__":
    unittest.main()
