import copy
import pickle
from itertools import product
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from qsplit import _core
from qsplit import qubo as qubo_module
from qsplit.conflict_resolution.local_search import nan_subqubo
from qsplit.halting_heuristic.stop import is_empty, vars_count
from qsplit.qubo import QUBO
from qsplit.refinement.propagation import condition_subproblem
from qsplit.splitting.split_recursive import split_leaves
from qsplit.splitting.split_recursive_graph import split_problem as split_graph


def test_strided_arrays_and_mutable_views():
    backing = np.arange(64.0).reshape(8, 8)
    raw = backing[::2, ::2].T
    ids = np.arange(8, dtype=np.int32)[::2]
    q = QUBO(raw, ids, ids[::-1])
    np.testing.assert_array_equal(q.mat, np.triu(raw + raw.T) - np.diag(np.diag(raw)))
    q.mat = np.triu(backing)[::2, ::2]
    q.rows_idx = np.arange(8)[::2]
    q.cols_idx = np.arange(8)[::2]
    assert vars_count(q) == 4
    q.mat[:] = 0
    assert is_empty(q)
    # The extension owns references, even when Python releases the source names.
    del backing, raw, ids
    q.mat[0, 1] = -5
    assert not is_empty(q)


@pytest.mark.parametrize("mutate", ["matrix", "indices", "size"])
def test_invalid_mutations_raise_instead_of_accessing_out_of_bounds(make_qubo, mutate):
    q = make_qubo(np.eye(2))
    if mutate == "matrix":
        q.mat = np.zeros((2, 3))
    elif mutate == "indices":
        q.rows_idx = np.array([0])
    else:
        q.problem_size = 10
    with pytest.raises((AssertionError, ValueError)):
        is_empty(q)


@pytest.mark.parametrize("protocol", [2, 4, 5])
def test_pre_port_pickle_dictionary_is_readable(monkeypatch, protocol):
    legacy_class = type("QUBO", (), {"__module__": "qsplit.qubo"})
    with monkeypatch.context() as patch:
        patch.setattr(qubo_module, "QUBO", legacy_class)
        old = legacy_class()
        old.__dict__.update(
            mat=np.array([[1.123456789123]]),
            rows_idx=np.array([17]),
            cols_idx=np.array([17]),
            offset=2.0,
            problem_size=1,
            solutions=pd.DataFrame({17: [1], "energy": [3.123456789123]}),
            macro_members={-1000: [17]},
        )
        serialized = pickle.dumps(old, protocol=protocol)
    restored = pickle.loads(serialized)
    assert isinstance(restored, _core.QUBO)
    assert restored.mat[0, 0] == old.mat[0, 0]  # Loading must not round again.
    assert restored.macro_members == old.macro_members
    pd.testing.assert_frame_equal(restored.solutions, old.solutions)


def test_pickle_and_deepcopy_preserve_recursive_metadata(make_qubo):
    q = make_qubo(np.triu(np.ones((6, 6))), offset=7)
    leaves = split_leaves(q, config={"CUT_DIM": 2})
    q.refinement_history = [7.0, 3.0]
    for restored in (copy.deepcopy(q), pickle.loads(pickle.dumps(q))):
        assert restored.refinement_history == [7.0, 3.0]
        assert restored.split_tree[0].problem_size == 6
        assert len(leaves) > 1
        restored.mat[0, 0] = 99
        assert q.mat[0, 0] == 1


def test_exact_conflicts_match_independent_exhaustive_objective(make_qubo):
    rng = np.random.default_rng(52)
    for _ in range(12):
        matrix = np.triu(rng.integers(-7, 8, size=(5, 5))).astype(float)
        ids = [40, 10, 70, 30, 20]
        q = make_qubo(matrix, ids=ids, offset=3)
        sample = {40: 1.0, 10: np.nan, 70: 0.0, 30: np.nan, 20: np.nan, "energy": 999.0}
        frame = pd.DataFrame([sample], index=["sample"])
        result = nan_subqubo(frame, q)
        energies = []
        for bits in product((0, 1), repeat=3):
            assignment = {**sample, **dict(zip([10, 20, 30], bits))}
            x = np.array([assignment[idx] for idx in ids])
            energies.append(float(x @ matrix @ x + 3))
        assert result.loc["sample", "energy"] == min(energies)
        assert frame.loc["sample", "energy"] == 999
        assert result.loc["sample", 40] == 1
        assert result.loc["sample", 70] == 0


def test_large_conflicts_call_python_solver_once_and_keep_offset(make_qubo):
    q = make_qubo(-np.eye(11), offset=5)
    samples = pd.DataFrame({**{i: [1.0] for i in range(11)}, "energy": [-11.0]})
    solve = Mock(return_value=samples)
    frame = pd.DataFrame(np.nan, index=["row"], columns=list(range(11)) + ["energy"])
    result = _core.resolve_conflicts(frame, q, solve, 10)
    solve.assert_called_once()
    assert isinstance(solve.call_args.args[0], QUBO)
    assert result.loc["row", "energy"] == -6


def test_conditioned_fractional_beliefs_match_expectation(make_qubo):
    q = make_qubo([[2, -4, 3], [0, 1, -2], [0, 0, -1]], ids=[20, 10, 30], offset=7)
    beliefs = {20: 0.25, 10: 0.75, 30: 0.5}
    sub = condition_subproblem(q, [30], beliefs)
    for bit in (0, 1):
        expected = 0.0
        for a, b in product((0, 1), repeat=2):
            weight = (0.25 if a else 0.75) * (0.75 if b else 0.25)
            x = np.array([a, b, bit])
            expected += weight * (x @ q.mat @ x + 7)
        assert bit * sub.mat[0, 0] + sub.offset == pytest.approx(expected)


def test_degenerate_graph_partition_still_terminates(monkeypatch, make_qubo):
    partition = Mock(side_effect=lambda *, nparts, adjacency: (0, [0] * len(adjacency)))
    monkeypatch.setattr("qsplit.splitting.split_recursive_graph.pymetis.part_graph", partition)
    q = make_qubo(np.triu(np.ones((7, 7))))
    subs = split_graph(q, config={"CUT_DIM": 2})
    assert sorted(idx for sub in subs for idx in sub.rows_idx) == list(range(7))
    assert all(sub.problem_size <= 2 for sub in subs)


@pytest.mark.parametrize("mode,expected", [("linear", 0), ("linear_belief_propagation", 1), ("recursive_graph", 1)])
def test_vote_ties_keep_existing_strategy_semantics(make_qubo, mode, expected):
    q = make_qubo([[-1]])
    subs = [make_qubo([[-1]]) for _ in range(2)]
    for sub, bit in zip(subs, (1, 0)):
        sub.solutions = pd.DataFrame({0: [bit], "energy": [0.0]})
    assert _core.aggregate_votes(subs, q, mode) is q
    assert q.solutions.loc[0, 0] == expected


@pytest.mark.parametrize("coefficient", [np.nan, np.inf])
def test_exact_conflicts_do_not_skip_nonfinite_coefficients(coefficient, make_qubo):
    q = make_qubo([[-1, coefficient], [0, -1]])
    result = nan_subqubo(pd.DataFrame({0: [np.nan], 1: [np.nan], "energy": [np.nan]}), q)
    # Every objective is NaN or +inf, so the existing solver retains its zero default.
    assert result.loc[0, [0, 1]].tolist() == [0, 0]
    assert np.isnan(result.loc[0, "energy"])
