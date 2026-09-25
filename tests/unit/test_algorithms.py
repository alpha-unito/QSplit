import importlib
from itertools import product

import numpy as np
import pandas as pd
import pytest

from qsplit.splitting import split_k_interactions, split_quadtree, split_recursive_graph


@pytest.mark.parametrize(
    "name", ["linear", "linear_belief_propagation", "k_interactions", "recursive_graph", "quadtree"]
)
def test_aggregators_respect_original_variable_order_padding_and_offset(name, make_qubo, assert_solution):
    qubo = make_qubo([[2, -3, 0], [0, -4, 0], [0, 0, 0]], ids=[20, 10, -1], offset=7)
    sub = make_qubo([[2, -3], [0, -4]], ids=[20, 10])
    sub.solutions = pd.DataFrame({20: [0], 10: [1], "energy": [-4.0]})
    aggregate = importlib.import_module(f"qsplit.aggregation.aggregate_{name}").aggregate_solutions
    assert_solution(aggregate([sub], qubo))
    assert qubo.solutions.energy.iloc[0] == 3


@pytest.mark.parametrize("cut", [1, 2, 4])
def test_interaction_neighborhoods_respect_cut_dimension(cut, monkeypatch, make_qubo):
    monkeypatch.setenv("CUT_DIM", str(cut))
    qubo = make_qubo(np.triu(np.ones((4, 4))), ids=[30, 10, 40, 20])
    subs = split_k_interactions.split_problem(qubo)
    assert len(subs) == 4
    for center, sub in zip(qubo.rows_idx, subs):
        assert center in sub.rows_idx
        assert sub.problem_size <= cut
        assert len(set(sub.rows_idx)) == sub.problem_size


@pytest.mark.parametrize("connected", [False, True])
def test_graph_partition_covers_every_real_variable_once(connected, make_qubo):
    matrix = np.triu(np.ones((6, 6))) if connected else np.eye(6)
    matrix[-1, :] = matrix[:, -1] = 0
    qubo = make_qubo(matrix, ids=[40, 20, 10, 30, 50, -1], offset=4)
    subs = split_recursive_graph.split_problem(qubo)
    assert sorted(idx for sub in subs for idx in sub.rows_idx) == [10, 20, 30, 40, 50]
    assert all(0 < sub.problem_size <= 2 for sub in subs)
    assert all(sub.offset == 4 for sub in subs)


def test_quadtree_macro_energy_equals_expanded_assignment(make_qubo, monkeypatch):
    monkeypatch.setenv("CUT_DIM", "4")
    monkeypatch.setenv("EXACT_RATIO", "0.5")
    matrix = np.triu(np.random.default_rng(17).integers(-4, 5, (6, 6)))
    qubo = make_qubo(matrix, ids=[60, 20, 10, 40, 30, 50], offset=5)
    for sub in split_quadtree.split_problem(qubo):
        for bits in product((0, 1), repeat=sub.problem_size):
            assignment = dict(zip(sub.rows_idx, bits))
            expanded = {key: value for key, value in assignment.items() if key >= 0}
            for macro, members in sub.macro_members.items():
                expanded.update(dict.fromkeys(members, assignment[macro]))
            x = np.array([expanded[idx] for idx in qubo.rows_idx])
            z = np.array(bits)
            assert z @ sub.mat @ z + sub.offset == pytest.approx(x @ qubo.mat @ x + qubo.offset)


def test_qubo_sanitization_preserves_binary_objective(make_qubo):
    raw = np.array([[1.0, -2.0, 3.0], [4.0, -5.0, 1.0], [-2.0, 3.0, 2.0]])
    qubo = make_qubo(raw)
    for bits in product((0, 1), repeat=3):
        x = np.array(bits)
        assert x @ qubo.mat @ x == pytest.approx(x @ raw @ x)


@pytest.mark.parametrize(
    "name", ["linear", "linear_belief_propagation", "k_interactions", "recursive_graph", "quadtree"]
)
@pytest.mark.parametrize("dummy_first", [False, True])
def test_dummy_assignments_do_not_override_valid_votes(name, dummy_first, make_qubo, assert_solution):
    from qsplit.adapters.dummy import solve

    qubo = make_qubo([[-2]], ids=[10], offset=7)
    empty = make_qubo([[0]], ids=[10])
    empty.solutions = solve(empty)
    valid = make_qubo([[-2]], ids=[10])
    valid.solutions = pd.DataFrame({10: [1], "energy": [-2.0]})
    subs = [empty, valid] if dummy_first else [valid, empty]
    aggregate = importlib.import_module(f"qsplit.aggregation.aggregate_{name}").aggregate_solutions
    result = aggregate(subs, qubo)
    assert_solution(result)
    assert result.solutions.iloc[0].to_dict() == {10: 1, "energy": 5}
