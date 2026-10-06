import json
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

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


@pytest.mark.parametrize("split_method", ["recursive", "linear", "k_interactions", "recursive_graph", "quadtree"])
@pytest.mark.parametrize("method", ["conditioned", "consensus", "mean_field", "soft_consensus"])
def test_distributed_rounds_match_local_native_controller(
    split_method, method, tmp_path, monkeypatch, exact_solver, assert_solution
):
    matrix = np.triu(np.random.default_rng(4).integers(-3, 4, (5, 5)))
    source = tmp_path / "input.csv"
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
        assert units <= 2 * 3
    result = load_qubo(state)
    assert_solution(result)
    assert result.refinement_history == expected.refinement_history
    assert result.solutions.energy.min() == expected.solutions.energy.min()
    if method == "conditioned":
        assert units == 3 * (len(result.refinement_history) - 1)
    else:
        assert units == len(result.refinement_history) - 1


@pytest.mark.parametrize("method", ["none", "conditioned", "mean_field", "soft_consensus"])
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


def test_conditioned_state_handles_duplicate_dataframe_indices_and_offset(make_qubo, exact_solver):
    qubo = make_qubo([[1, -3], [0, 1]], ids=[30, 10], offset=7)
    qubo.instance_id = "case"
    qubo.solutions = pd.DataFrame({30: [0, 1], 10: [0, 0], "energy": [7, 8]}, index=[5, 5])
    refine.initialize(qubo, [qubo], method="conditioned", loops=1, block_size=2, aggregate_method="linear")
    sub = refine.prepare(qubo, [qubo])[0]
    sub.solutions = exact_solver(sub)
    refine.update(qubo, [sub])
    assert qubo.refinement_history == [7, 6]
    assert qubo.solutions.iloc[0].tolist() == [1, 1, 6]
