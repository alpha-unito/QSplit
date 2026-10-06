"""Cross-pipeline refinements, independent composition and native strategy invariants."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from qsplit import _core, configuration, local_runner
from qsplit.aggregation.aggregate_linear import aggregate_solutions as votes
from qsplit.aggregation.aggregate_linear_belief_propagation import aggregate_solutions as bp
from qsplit.aggregation.aggregate_recursive import aggregate_leaves
from qsplit.aggregation.aggregate_recursive_graph import aggregate_solutions as graph_votes
from qsplit.refinement import refine_result
from qsplit.refinement.refine_mean_field import refine_problems as mean_field
from qsplit.refinement.refine_soft_consensus import refine_problems as soft_consensus
from qsplit.splitting.split_k_interactions import split_problem as interactions
from qsplit.splitting.split_linear import split_problem as linear
from qsplit.splitting.split_quadtree import split_problem as quadtree
from qsplit.splitting.split_recursive import split_leaves
from qsplit.splitting.split_recursive_graph import split_problem as graph

PIPELINES = ["iterative", "quadtree", "recursive", "interactions", "graph_partitioning"]
METHODS = ["conditioned", "consensus", "mean_field", "soft_consensus"]


@pytest.mark.parametrize("pipeline", PIPELINES)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("size", [1, 5])
def test_every_refinement_preserves_complete_best_global_solution(
    pipeline, method, size, make_qubo, exact_solver, assert_solution, monkeypatch
):
    monkeypatch.setattr(local_runner, "solve", exact_solver)
    sampler = getattr(local_runner, f"qsplit_sampler_refined_{pipeline}")
    ids = np.random.default_rng(size).permutation(np.arange(size) * 10 + 10)
    matrix = np.triu(np.random.default_rng(4).integers(-3, 4, (size, size)))
    qubo = make_qubo(matrix, ids=ids, offset=7)
    result = sampler(qubo, config={"CUT_DIM": 2, "REFINEMENT_LOOPS": 2, "REFINEMENT_METHOD": method})
    assert_solution(result)
    assert 1 <= len(result.refinement_history) <= 3
    assert result.solutions.energy.min() == min(result.refinement_history)
    assert result.solutions.energy.min() <= result.refinement_history[0]
    if method == "conditioned":
        assert np.all(np.diff(result.refinement_history) <= 0)


@pytest.mark.parametrize("pipeline", PIPELINES)
@pytest.mark.parametrize("method", METHODS)
def test_constant_objectives_do_not_call_backend(pipeline, method, make_qubo, monkeypatch, assert_solution):
    def fail(_):
        pytest.fail("A constant problem must not call the solver")

    monkeypatch.setattr(local_runner, "solve", fail)
    result = getattr(local_runner, f"qsplit_sampler_refined_{pipeline}")(
        make_qubo(np.zeros((5, 5)), offset=7),
        config={"CUT_DIM": 2, "REFINEMENT_LOOPS": 2, "REFINEMENT_METHOD": method},
    )
    assert_solution(result)
    assert result.solutions.energy.min() == 7


@pytest.mark.parametrize("pipeline", PIPELINES)
@pytest.mark.parametrize("loops", [0, -1])
def test_disabled_methods_delegate_to_legacy(pipeline, loops, make_qubo, monkeypatch):
    sentinel = object()
    calls = []
    monkeypatch.setattr(local_runner, f"qsplit_sampler_{pipeline}", lambda q: calls.append(q) or sentinel)
    qubo = make_qubo([[1]])
    result = getattr(local_runner, f"qsplit_sampler_refined_{pipeline}")(
        qubo,
        refinement=lambda *_: pytest.fail("Disabled refinement"),
        config={"REFINEMENT_LOOPS": loops, "REFINEMENT_METHOD": "invalid", "REFINEMENT_PATIENCE": "invalid"},
    )
    assert result is sentinel and calls == [qubo]


@pytest.mark.parametrize("split", [linear, interactions, quadtree, graph, split_leaves])
@pytest.mark.parametrize("aggregate", [votes, bp, graph_votes])
@pytest.mark.parametrize("method", ["mean_field", "soft_consensus"])
def test_independent_split_aggregate_refinement(
    split, aggregate, method, make_qubo, exact_solver, monkeypatch, assert_solution
):
    monkeypatch.setattr(local_runner, "solve", exact_solver)
    result = local_runner.qsplit_sampler(
        make_qubo(np.triu(-np.ones((5, 5))), offset=4),
        split=split,
        aggregate=aggregate,
        refinement_aggregate=aggregate,
        config={"CUT_DIM": 2, "REFINEMENT_LOOPS": 1, "REFINEMENT_METHOD": method},
    )
    assert_solution(result)
    assert result.solutions.energy.min() <= result.refinement_history[0]


def test_recursive_tree_initialization_with_flat_refinement(make_qubo, exact_solver, monkeypatch, assert_solution):
    monkeypatch.setattr(local_runner, "solve", exact_solver)
    result = local_runner.qsplit_sampler(
        make_qubo(np.triu(-np.ones((5, 5))), offset=4),
        split=split_leaves,
        aggregate=aggregate_leaves,
        config={"CUT_DIM": 2, "REFINEMENT_LOOPS": 2, "REFINEMENT_METHOD": "mean_field"},
    )
    assert_solution(result)
    assert len(result.refinement_history) == 2


@pytest.mark.parametrize("method", METHODS)
def test_refine_any_incumbent_without_split_templates(method, make_qubo, exact_solver, assert_solution):
    qubo = make_qubo([[1, -3], [0, 1]], ids=[30, 10], offset=5)
    qubo.solutions = pd.DataFrame({30: [0], 10: [0], "energy": [-999]})
    result = refine_result(
        qubo,
        solve=exact_solver,
        config={"CUT_DIM": 2, "REFINEMENT_LOOPS": 1, "REFINEMENT_METHOD": method},
    )
    assert_solution(result)
    assert result.refinement_history == [5, 4]


@pytest.mark.parametrize("strategy", [mean_field, soft_consensus])
def test_native_strategies_rebuild_bounded_real_windows(strategy, make_qubo, exact_solver):
    qubo = make_qubo(np.triu(np.arange(25).reshape(5, 5) - 9), ids=[50, 10, 30, 40, 20], offset=7)
    qubo.solutions = exact_solver(qubo)
    template = deepcopy(qubo)
    template.solutions = None
    before = template.mat.copy()
    with configuration.use({"CUT_DIM": 2, "REFINEMENT_STRENGTH": 0.2}):
        first = strategy([template], qubo)
        repeated = strategy(first, qubo)
    assert [sub.problem_size for sub in first] == [2, 2, 1]
    for a, b in zip(first, repeated):
        assert np.array_equal(a.rows_idx, a.cols_idx)
        assert a.solutions is None
        np.testing.assert_allclose(a.mat, b.mat)
        assert a.offset == pytest.approx(b.offset)
    np.testing.assert_array_equal(template.mat, before)


def test_soft_consensus_reuses_native_macro_objective_and_evidence(make_qubo, exact_solver):
    qubo = make_qubo(np.triu(-np.ones((5, 5))), offset=7)
    qubo.solutions = exact_solver(qubo)
    subs = quadtree(qubo, config={"CUT_DIM": 2})
    for sub in subs:
        sub.solutions = exact_solver(sub)
    expected = _core.refine_quadtree(subs, qubo, 0.2)
    actual = _core.refine_soft_consensus(subs, qubo, 0.2, 2)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a.mat, b.mat)
        assert a.offset == pytest.approx(b.offset)
        assert a.macro_members == b.macro_members


@pytest.mark.parametrize("method", ["conditioned", "mean_field", "soft_consensus"])
def test_recursive_logical_expansion_initialization_is_preserved(method, make_qubo, exact_solver, monkeypatch):
    monkeypatch.setattr(local_runner, "solve", exact_solver)
    monkeypatch.setattr(local_runner, "LOGICAL_EXPANSION", True)
    qubo = make_qubo(np.triu(-np.ones((5, 5))), offset=7)
    legacy = local_runner.qsplit_sampler_recursive(deepcopy(qubo))
    result = local_runner.qsplit_sampler_refined_recursive(
        qubo,
        config={"CUT_DIM": 2, "REFINEMENT_LOOPS": 1, "REFINEMENT_METHOD": method},
    )
    assert result.refinement_history[0] == legacy.solutions.energy.min()


def test_post_refinement_validates_missing_incumbent_and_disabled_path(make_qubo, exact_solver):
    qubo = make_qubo([[1]])
    assert refine_result(qubo, solve=exact_solver, config={"REFINEMENT_LOOPS": 0}) is qubo
    with pytest.raises(ValueError, match="complete global solution"):
        refine_result(qubo, solve=exact_solver, config={"CUT_DIM": 2, "REFINEMENT_LOOPS": 1})


def test_native_macro_templates_validate_shape_and_actual_size(make_qubo, exact_solver):
    qubo = make_qubo(-np.eye(5))
    qubo.solutions = exact_solver(qubo)
    sub = quadtree(qubo, config={"CUT_DIM": 4})[0]
    sub.problem_size = 1  # Mutable metadata must not bypass the actual window limit.
    with pytest.raises(ValueError, match="problem_size"):
        _core.refine_soft_consensus([sub], qubo, 0.1, 2)
    sub.problem_size = 4
    with pytest.raises(ValueError, match="exceeds"):
        _core.refine_soft_consensus([sub], qubo, 0.1, 2)
    sub.cols_idx = sub.cols_idx[::-1].copy()
    with pytest.raises(ValueError, match="principal"):
        _core.refine_soft_consensus([sub], qubo, 0.1, 4)
