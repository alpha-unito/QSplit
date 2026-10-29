import json
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from qsplit.cwl.cli import aggregate, refine, split
from qsplit.cwl.cli.utils import load_qubo, save_qubo
from qsplit.refinement import refine_result


def invoke(monkeypatch, directory, module, *args):
    directory.mkdir(exist_ok=True)
    monkeypatch.chdir(directory)
    monkeypatch.setattr(sys, "argv", [module.__name__, *map(str, args)])
    module.main()


def solve_files(sources, directory, solver):
    directory.mkdir(exist_ok=True)
    paths = []
    for index, source in enumerate(sources):
        qubo = load_qubo(source)
        qubo.solutions, qubo.backend = solver(qubo), "independent-exact"
        path = directory / f"{index:06d}.pkl"
        save_qubo(path, qubo)
        paths.append(path)
    return paths


@pytest.mark.parametrize("sparse_input", [False, True])
@pytest.mark.parametrize("split_method", ["recursive", "linear", "k_interactions", "recursive_graph", "quadtree"])
@pytest.mark.parametrize("method", ["consensus", "mean_field", "soft_consensus"])
def test_distributed_rounds_match_local_native_controller(
    split_method, method, sparse_input, tmp_path, monkeypatch, exact_solver, assert_solution
):
    matrix = np.triu(np.random.default_rng(4).integers(-3, 4, (5, 5)))
    source = tmp_path / ("input.npz" if sparse_input else "input.csv")
    if sparse_input:
        sparse.save_npz(source, sparse.csr_array(matrix))
    else:
        np.savetxt(source, matrix, delimiter=",")
    start = tmp_path / "split"
    invoke(monkeypatch, start, split, "--input-matrix", source, "--cut-dim", 2, "--split-method", split_method)
    paths = solve_files(sorted((start / "planned/parallel").glob("*.pkl")), tmp_path / "initial-solves", exact_solver)
    paths += sorted((start / "solved_dummy").glob("*.pkl"))
    initial = tmp_path / "aggregate"
    invoke(
        monkeypatch,
        initial,
        aggregate,
        "--input-qubo",
        start / "initial_qubo.pkl",
        "--tree-file",
        start / "tree.json",
        "--solved-list",
        *paths,
        "--skip-local-refinement",
    )
    qubo = load_qubo(initial / "aggregate_qubo.pkl")
    subs = qubo.refinement_subproblems
    expected = refine_result(
        deepcopy(qubo),
        solve=exact_solver,
        subproblems=subs,
        config={"CUT_DIM": 2, "REFINEMENT_LOOPS": 2, "REFINEMENT_METHOD": method},
    )
    initialized = tmp_path / "initialize"
    invoke(
        monkeypatch,
        initialized,
        refine,
        "--phase",
        "initialize",
        "--input-state",
        initial / "aggregate_qubo.pkl",
        "--method",
        method,
        "--loops",
        2,
        "--cut-dim",
        2,
    )
    state = initialized / "state.pkl"
    units = 0
    while json.loads(state.with_name("control.json").read_text())["continue"]:
        prepared = tmp_path / f"prepare-{units}"
        invoke(monkeypatch, prepared, refine, "--phase", "prepare", "--input-state", state)
        sources = sorted((prepared / "subproblems").glob("*.pkl"))
        assert all(load_qubo(path).problem_size <= 2 for path in sources)
        solved = solve_files(sources, tmp_path / f"solve-{units}", exact_solver)
        solved += sorted((prepared / "solved_constant").glob("*.pkl"))
        updated = tmp_path / f"update-{units}"
        invoke(
            monkeypatch,
            updated,
            refine,
            "--phase",
            "update",
            "--input-state",
            prepared / "state.pkl",
            "--solved-list",
            *solved,
        )
        state = updated / "state.pkl"
        units += 1
        assert units <= 2
    result = load_qubo(state)
    assert_solution(result)
    assert result.refinement_history == expected.refinement_history
    assert result.solutions.energy.min() == expected.solutions.energy.min()
    assert units == len(result.refinement_history) - 1


@pytest.mark.parametrize("method", ["none", "mean_field", "soft_consensus"])
def test_zero_objective_and_disabled_refinement(method, tmp_path, monkeypatch, exact_solver):
    source = tmp_path / "zero.csv"
    np.savetxt(source, np.zeros((3, 3)), delimiter=",")
    start = tmp_path / "split"
    invoke(monkeypatch, start, split, "--input-matrix", source, "--cut-dim", 2, "--split-method", "linear")
    assert not list((start / "planned/parallel").glob("*.pkl"))
    paths = list((start / "solved_dummy").glob("*.pkl"))
    initial = tmp_path / "aggregate"
    invoke(
        monkeypatch,
        initial,
        aggregate,
        "--input-qubo",
        start / "initial_qubo.pkl",
        "--tree-file",
        start / "tree.json",
        "--solved-list",
        *paths,
        "--skip-local-refinement",
    )
    initialized = tmp_path / "initialize"
    invoke(
        monkeypatch,
        initialized,
        refine,
        "--phase",
        "initialize",
        "--input-state",
        initial / "aggregate_qubo.pkl",
        "--method",
        method,
        "--loops",
        1,
        "--cut-dim",
        2,
    )
    state = initialized / "state.pkl"
    count = 0
    while load_qubo(state).refinement_state["active"]:
        prepared, updated = tmp_path / f"prepare-{count}", tmp_path / f"update-{count}"
        invoke(monkeypatch, prepared, refine, "--phase", "prepare", "--input-state", state)
        assert not list((prepared / "subproblems").glob("*.pkl"))
        paths = list((prepared / "solved_constant").glob("*.pkl"))
        invoke(
            monkeypatch,
            updated,
            refine,
            "--phase",
            "update",
            "--input-state",
            prepared / "state.pkl",
            "--solved-list",
            *paths,
        )
        state = updated / "state.pkl"
        count += 1
        assert count <= 2
    assert load_qubo(state).solutions.energy.min() == 0
    assert (count == 0) == (method == "none")


def test_transition_rejects_foreign_duplicate_and_stale_samples(tmp_path, make_qubo):
    q = make_qubo([[1]])
    q.instance_id, q.node_id = "first", "refine_0"
    q.solutions = pd.DataFrame({0: [0], "energy": [0]})
    path = tmp_path / "solved.pkl"
    save_qubo(path, q)
    for paths, instance, expected in [
        ([path], "second", None),
        ([path, path], "first", None),
        ([path], "first", {"refine_1": [[0], [0]]}),
        ([path], "first", {"refine_0": [[1], [1]]}),
    ]:
        with pytest.raises(ValueError):
            refine.load_samples(paths, instance, expected)


@pytest.mark.parametrize("method", ["linear", "quadtree", "recursive"])
def test_invalid_initial_combinations_fail_before_solving(method, tmp_path, monkeypatch):
    np.savetxt(tmp_path / "input.csv", np.eye(3), delimiter=",")
    with pytest.raises(ValueError, match="incompatible"):
        invoke(
            monkeypatch,
            tmp_path / "split",
            split,
            "--input-matrix",
            tmp_path / "input.csv",
            "--cut-dim",
            2,
            "--split-method",
            method,
            "--aggregate-method",
            "k_interactions",
        )


@pytest.mark.parametrize("method", ["consensus", "mean_field", "soft_consensus"])
def test_prepare_builds_entire_round_before_update(method, make_qubo, exact_solver, assert_solution):
    qubo = make_qubo(-np.eye(5), ids=[30, 10, 50, 20, 40], offset=7)
    qubo.instance_id = "case"
    qubo.solutions = pd.DataFrame({**{idx: [0, 0] for idx in qubo.rows_idx}, "energy": [7, 7]}, index=[5, 5])
    templates = [deepcopy(qubo)]
    refine.initialize(qubo, templates, method=method, loops=1, block_size=2, aggregate_method="linear")

    blocks = refine.prepare(qubo, templates)
    assert len(blocks) == 3
    assert sorted(idx for sub in blocks for idx in sub.rows_idx) == sorted(qubo.rows_idx)
    assert all(sub.solutions is None and sub.problem_size <= 2 for sub in blocks)
    assert len(qubo.refinement_state["pending"]) == 3
    assert qubo.refinement_state["rounds"] == 0
    assert qubo.refinement_history == [7]
    assert (qubo.solutions.energy == 7).all()

    for sub in blocks:
        sub.solutions = exact_solver(sub)
    refine.update(qubo, list(reversed(blocks)))
    assert_solution(qubo)
    assert qubo.refinement_history == [7, 2]
    assert qubo.refinement_state["rounds"] == 1
    assert not qubo.refinement_state["active"]


def test_removed_refinement_is_rejected_in_settings_and_saved_state(make_qubo, exact_solver):
    qubo = make_qubo([[-1]])
    qubo.instance_id = "case"
    qubo.solutions = exact_solver(qubo)
    with pytest.raises(ValueError, match="Unknown refinement method"):
        refine.initialize(qubo, [qubo], method="conditioned", loops=1, block_size=2, aggregate_method="linear")
    refine.initialize(qubo, [qubo], method="mean_field", loops=1, block_size=2, aggregate_method="linear")
    qubo.refinement_state["options"]["method"] = "conditioned"
    with pytest.raises(ValueError, match="Unknown refinement method in saved state"):
        refine.prepare(qubo, [qubo])
    with pytest.raises(ValueError, match="Unknown refinement method in saved state"):
        refine.update(qubo, [qubo])
