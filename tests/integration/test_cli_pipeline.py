"""Exercise the same entry points used by CWL, with real files and samplers."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qsplit.cwl.cli import (
    aggregate,
    collect_dataset_results,
    dataset_prepare,
    persist_instance_solution,
    scatter,
    split,
)
from qsplit.cwl.cli.utils import load_qubo, save_qubo


def invoke(monkeypatch, module, *args):
    monkeypatch.setattr(sys, "argv", [module.__name__, *map(str, args)])
    module.main()


@pytest.mark.parametrize("size,sparse", [(3, False), (4, False), (4, True), (1, False)])
def test_split_solve_aggregate_pipeline(size, sparse, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    matrix = -np.eye(size) if sparse else np.triu(-np.ones((size, size)))
    np.savetxt("case.csv", matrix, delimiter=",")
    invoke(
        monkeypatch, split, "--input-matrix", "case.csv", "--cut-dim", 2, *(["--enable-sparse-check"] if sparse else [])
    )
    tree = json.loads(Path("tree.json").read_text())
    solved = list(Path("solved_dummy").glob("*.pkl"))
    planned = list(Path("planned/parallel").glob("*.pkl"))
    assert not list(Path("planned/iqm").glob("*.pkl"))
    for i, sub in enumerate(planned):
        result = Path(f"solved_{i}.pkl")
        invoke(monkeypatch, scatter, "--input-qubo", sub, "--output-qubo", result)
        assert load_qubo(result).backend == "dwave"
        solved.append(result)
    leaves = {node for node, spec in tree["nodes"].items() if not spec["children"]}
    assert {load_qubo(path).node_id for path in solved} == leaves
    invoke(
        monkeypatch, aggregate, "--input-qubo", "initial_qubo.pkl", "--tree-file", "tree.json", "--solved-list", *solved
    )
    df = pd.read_csv("solutions.csv", dtype={"bitstring": str})
    roots = df[(df.node_id == "root") & (df.backend == "aggregate")]
    assert not roots.empty
    for row in roots.itertuples():
        assert len(row.bitstring) == size
        x = np.array(list(row.bitstring), dtype=int)
        assert row.energy == pytest.approx(x @ matrix @ x)


def test_missing_or_foreign_leaf_fails_explicitly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    np.savetxt("case.csv", -np.eye(2), delimiter=",")
    invoke(monkeypatch, split, "--input-matrix", "case.csv")
    qubo = load_qubo("initial_qubo.pkl")
    qubo.instance_id = "another-instance"
    qubo.solutions = pd.DataFrame({0: [1], 1: [1], "energy": [-2]})
    save_qubo("foreign.pkl", qubo)
    with pytest.raises(RuntimeError, match="missing_solved_leaf_nodes"):
        invoke(
            monkeypatch,
            aggregate,
            "--input-qubo",
            "initial_qubo.pkl",
            "--tree-file",
            "tree.json",
            "--solved-list",
            "foreign.pkl",
        )
    assert not Path("solutions.csv").exists()


def test_dataset_prepare_persist_collect_and_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = tmp_path / "store"
    dataset = tmp_path / "dataset.jsonl"
    records = [
        {"id": "case A", "dim": 2, "qubo_mat": [[0, 0, -1], [1, 1, -2]]},
        {"originale_index": 9, "dim": 3, "qubo_mat": [[0, 2, -1]]},
    ]
    dataset.write_text("bad json\n\n" + "\n".join(json.dumps(r) for r in records))
    args = ("--dataset-jsonl", dataset, "--solutions-dir", store)
    invoke(monkeypatch, dataset_prepare, *args)
    manifest = json.loads(Path("dataset_manifest.json").read_text())
    assert manifest["count"] == manifest["scheduled_count"] == 2
    assert [r["safe_id"] for r in manifest["items"]] == ["case_A", "9"]
    np.testing.assert_array_equal(np.loadtxt("dataset_matrices/000000_case_A.csv", delimiter=","), [[-1, 0], [0, -2]])
    contents = "node_id,backend,bitstring,energy\nroot,aggregate,11,-3\n"
    Path("candidate.csv").write_text(contents)
    invoke(
        monkeypatch,
        persist_instance_solution,
        "--input-solution",
        "candidate.csv",
        "--input-matrix",
        "000000_000001_case_A.csv",
        "--solutions-dir",
        store,
    )
    assert (store / "solutions_case_A.csv").read_text() == contents
    assert Path("persisted_solution.csv").read_text() == contents
    invoke(
        monkeypatch, collect_dataset_results, "--dataset-manifest", "dataset_manifest.json", "--solutions-dir", store
    )
    result = json.loads(Path("dataset_results_manifest.json").read_text())
    assert result["resolved_solutions"] == 1
    assert [r["status"] for r in result["items"]] == ["ok", "missing"]
    invoke(monkeypatch, dataset_prepare, *args)
    resumed = json.loads(Path("dataset_manifest.json").read_text())
    assert resumed["scheduled_count"] == 1
    assert resumed["items"][0]["already_solved"] is True
    assert resumed["items"][0]["matrix_csv"] is None
    assert len(list(Path("dataset_matrices").glob("*.csv"))) == 1


def test_dataset_legacy_solution_name_and_limit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "solutions_000000_case.csv").write_text("a,b,c,d\nroot,aggregate,11,-3\n")
    Path("data.jsonl").write_text("\n".join(json.dumps({"id": name, "dim": 2}) for name in ["case", "other"]))
    invoke(
        monkeypatch, dataset_prepare, "--dataset-jsonl", "data.jsonl", "--solutions-dir", tmp_path, "--max-instances", 1
    )
    manifest = json.loads(Path("dataset_manifest.json").read_text())
    assert manifest["count"] == 1 and manifest["scheduled_count"] == 0
    assert Path("solutions_case.csv").exists()


@pytest.mark.parametrize("contents", ["", "invalid JSON\n", '{"dim": 0}\n'])
def test_dataset_without_valid_records_fails(contents, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("data.jsonl").write_text(contents)
    with pytest.raises(SystemExit, match="No valid records"):
        invoke(monkeypatch, dataset_prepare, "--dataset-jsonl", "data.jsonl", "--solutions-dir", tmp_path / "store")


def test_workspace_recovery_uses_only_matching_instance_and_indices(tmp_path, make_qubo):
    root = tmp_path / "streamflow_workdir"
    specs = {"root_0": ((0, 1), (0, 1))}
    paths = []
    for folder, instance, ids in [
        ("valid", "case", [0, 1]),
        ("foreign", "other", [0, 1]),
        ("wrong_indices", "case", [2, 3]),
    ]:
        directory = root / folder
        directory.mkdir(parents=True)
        qubo = make_qubo(-np.eye(2), ids=ids)
        qubo.node_id, qubo.instance_id = "root_0", instance
        qubo.solutions = pd.DataFrame({ids[0]: [1], ids[1]: [1], "energy": [-2]})
        path = directory / "solved.pkl"
        save_qubo(path, qubo)
        paths.append(path)
    assert aggregate._workspace_roots_from_paths(paths) == [root]
    recovered = aggregate._discover_solved_for_instance([root], "case", {"root_0"}, set(), specs)
    assert len(recovered) == 1 and recovered[0][0] == paths[0]
    assert aggregate._discover_solved_for_instance([root], "case", {"root_0"}, {paths[0]}, specs) == []
