"""Black-box CLI and actual CWL execution. No cluster configurations are used."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def run_command(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    # CWL may launch each command in a different working directory. Coverage's
    # subprocess patch propagates these absolute paths to every child process.
    if "COVERAGE_RCFILE" in os.environ:
        monkeypatch.setenv("COVERAGE_RCFILE", str(Path(os.environ["COVERAGE_RCFILE"]).resolve()))

    def run(*command, cwd=tmp_path, timeout=180):
        result = subprocess.run(
            list(map(str, command)), cwd=cwd, env=os.environ.copy(), text=True, capture_output=True, timeout=timeout
        )
        assert result.returncode == 0, f"Command: {command}\n{result.stdout}\n{result.stderr}"
        return result

    return run


def check_result(path, matrix):
    frame = pd.read_csv(path, dtype={"bitstring": str})
    roots = frame[(frame.node_id == "root") & (frame.backend == "aggregate")]
    assert not roots.empty
    for row in roots.itertuples():
        assert len(row.bitstring) == len(matrix)
        assert set(row.bitstring) <= {"0", "1"}
        x = np.array(list(row.bitstring), dtype=int)
        assert row.energy == pytest.approx(x @ matrix @ x)


def test_installed_commands_split_scatter_aggregate(tmp_path, run_command):
    matrix = np.triu(-np.ones((3, 3)))
    np.savetxt(tmp_path / "input.csv", matrix, delimiter=",")
    run_command("cli_split", "--input-matrix", "input.csv", "--cut-dim", "2")
    solved = list((tmp_path / "solved_dummy").glob("*.pkl"))
    for index, source in enumerate(sorted((tmp_path / "planned/parallel").glob("*.pkl"))):
        output = tmp_path / f"solved-{index}.pkl"
        run_command("cli_scatter", "--input-qubo", source, "--output-qubo", output)
        solved.append(output)
    run_command(
        "cli_aggregate", "--input-qubo", "initial_qubo.pkl", "--tree-file", "tree.json", "--solved-list", *solved
    )
    check_result(tmp_path / "solutions.csv", matrix)


@pytest.mark.streamflow
def test_actual_cwl_dataset_workflow_and_resume(tmp_path, run_command):
    pytest.importorskip("streamflow.main")
    matrix = np.triu(-np.ones((3, 3)))
    records = [
        {
            "id": name,
            "dim": len(matrix),
            "qubo_mat": [[i, j, float(matrix[i, j])] for i in range(3) for j in range(i, 3)],
        }
        for name in ["first", "second"]
    ]
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("\n".join(json.dumps(record) for record in records))
    settings = tmp_path / "settings.json"
    store = tmp_path / "store"
    settings.write_text(
        json.dumps(
            {
                "dataset": {"class": "File", "path": str(dataset)},
                "cut_dim": 2,
                "enable_iqm": False,
                "enable_quantinuum_h2": False,
                "enable_quantinuum_h2e": False,
                "solutions_store_dir": str(store),
            }
        )
    )
    config = tmp_path / "streamflow.json"
    config.write_text(
        json.dumps(
            {
                "version": "v1.0",
                "workflows": {
                    "local-test": {
                        "type": "cwl",
                        "config": {"file": str(REPO / "streamflow/cwl/main.cwl"), "settings": str(settings)},
                        "bindings": [
                            {"step": "/qsplit_instances/parallelize", "target": {"deployment": "cpu-simulator"}}
                        ],
                    }
                },
                "deployments": {
                    "local": {"type": "local", "workdir": str(tmp_path / "work"), "config": {}},
                    "cpu-simulator": {
                        "type": "qsplit.quantum_connector",
                        "wraps": "local",
                        "config": {
                            "provider": "dwave",
                            "providerPool": ["dwave"],
                            "maxConcurrentJobs": {"dwave": 2},
                            "providerEnvMap": {"dwave": {"PATH": os.environ["PATH"]}},
                        },
                    },
                },
                "database": {"type": "sqlite", "config": {"connection": str(tmp_path / "streamflow.db")}},
            }
        )
    )
    run_command("streamflow", "run", "--outdir", tmp_path / "output", config)
    assert len(list(store.glob("solutions_*.csv"))) == 2
    for path in store.glob("solutions_*.csv"):
        check_result(path, matrix)
    original = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in store.glob("*.csv")}
    run_command("streamflow", "run", "--outdir", tmp_path / "resumed", config)
    assert {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in store.glob("*.csv")} == original
    manifests = list((tmp_path / "resumed").rglob("dataset_results_manifest.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())["resolved_solutions"] == 2
