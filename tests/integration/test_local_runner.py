from copy import deepcopy

import numpy as np
import pytest

from qsplit import local_runner
from qsplit.aggregation.aggregate_recursive import aggregate_leaves
from qsplit.splitting.split_recursive import split_leaves

SAMPLERS = [
    local_runner.qsplit_sampler_recursive,
    local_runner.qsplit_sampler_iterative,
    local_runner.qsplit_sampler_interactions,
    local_runner.qsplit_sampler_graph_partitioning,
    local_runner.qsplit_sampler_quadtree,
]


@pytest.mark.parametrize("sampler", SAMPLERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("size", [3, 4, 6])
def test_decomposition_solution_has_correct_global_energy(
    sampler, size, make_qubo, exact_solver, assert_solution, monkeypatch
):
    monkeypatch.setattr(local_runner, "solve", exact_solver)
    matrix = np.triu(np.random.default_rng(size).integers(-4, 5, (size, size))).astype(float)
    ids = np.random.default_rng(size).permutation(np.arange(size) + 10)
    qubo = make_qubo(matrix, ids=ids, offset=7)
    original = qubo.mat.copy()
    result = sampler(qubo)
    assert_solution(result)
    np.testing.assert_array_equal(result.mat[:size, :size], original)
    # Decomposition is heuristic; it may not reach the exact optimum.
    optimum = exact_solver(make_qubo(matrix, ids=ids, offset=7)).energy.min()
    assert result.solutions.energy.min() >= optimum


@pytest.mark.parametrize("sampler", SAMPLERS, ids=lambda f: f.__name__)
def test_real_annealer_through_every_runner(sampler, make_qubo, assert_solution):
    result = sampler(make_qubo(np.triu(-np.ones((4, 4)))))
    assert_solution(result)


@pytest.mark.parametrize("logical,bp", [(True, False), (False, True)])
def test_optional_aggregation_modes(logical, bp, monkeypatch, make_qubo, exact_solver, assert_solution):
    monkeypatch.setattr(local_runner, "LOGICAL_EXPANSION", logical)
    monkeypatch.setattr(local_runner, "BP", bp)
    monkeypatch.setattr(local_runner, "solve", exact_solver)
    qubo = make_qubo(np.triu(-np.ones((4, 4))))
    sampler = local_runner.qsplit_sampler_recursive if logical else local_runner.qsplit_sampler_iterative
    assert_solution(sampler(qubo))


def test_recursive_leaf_api_matches_recursive_runner(make_qubo, exact_solver, monkeypatch, assert_solution):
    monkeypatch.setattr(local_runner, "solve", exact_solver)
    qubo = make_qubo(np.triu(-np.ones((5, 5))))
    expected = local_runner.qsplit_sampler_recursive(deepcopy(qubo))
    leaves = split_leaves(qubo)
    assert len(leaves) > 1
    for leaf in leaves:
        leaf.solutions = exact_solver(leaf)
    result = aggregate_leaves(leaves, qubo)
    assert_solution(result)
    assert result.solutions.energy.min() == expected.solutions.energy.min()


@pytest.mark.parametrize("method", ["conditioned", "consensus"])
@pytest.mark.parametrize(
    "sampler", [local_runner.qsplit_sampler_refined_iterative, local_runner.qsplit_sampler_refined_quadtree]
)
def test_refinement_with_real_annealing(method, sampler, monkeypatch, make_qubo, assert_solution):
    monkeypatch.setenv("REFINEMENT_METHOD", method)
    monkeypatch.setenv("REFINEMENT_LOOPS", "2")
    result = sampler(make_qubo(np.triu(-np.ones((4, 4))), offset=3))
    assert_solution(result)
    assert 1 <= len(result.refinement_history) <= 3
    assert result.solutions.energy.min() == min(result.refinement_history)


@pytest.mark.parametrize("diagonal", [[0, 0, 0], [0, -2, 1]])
def test_graph_runner_handles_empty_partitions(diagonal, monkeypatch, make_qubo, exact_solver, assert_solution):
    monkeypatch.setenv("CUT_DIM", "1")
    solved_ids = []

    def solve(sub):
        solved_ids.extend(sub.rows_idx)
        return exact_solver(sub)

    monkeypatch.setattr(local_runner, "solve", solve)
    result = local_runner.qsplit_sampler_graph_partitioning(make_qubo(np.diag(diagonal), ids=[30, 10, 20], offset=7))
    assert_solution(result)
    assert result.solutions.iloc[0].to_dict() == {
        10: int(diagonal[1] < 0),
        20: 0,
        30: 0,
        "energy": 7 + min(diagonal[1], 0),
    }
    assert sorted(solved_ids) == ([10, 20] if any(diagonal) else [])
