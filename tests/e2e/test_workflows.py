import json
import os
import signal
import subprocess
import sys
import sysconfig
from itertools import count
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qsplit.cwl.cli.utils import load_qubo

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def run_command(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", sysconfig.get_path("scripts") + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("PYTHONUNBUFFERED", "1")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    if "COVERAGE_RCFILE" in os.environ:
        monkeypatch.setenv("COVERAGE_RCFILE", str(Path(os.environ["COVERAGE_RCFILE"]).resolve()))

    command_ids = count(1)

    def run(*command, cwd=tmp_path, timeout=180):
        log_path = tmp_path / f"{next(command_ids):02d}-{Path(str(command[0])).name}.log"
        with log_path.open("w") as log:
            log.write(f"Command: {command}\n")
            log.flush()
            with subprocess.Popen(
                list(map(str, command)),
                cwd=cwd,
                env=os.environ.copy(),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            ) as process:
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
                    pytest.fail(
                        f"Command timed out after {timeout}s. Full log: {log_path}\n"
                        f"{log_path.read_text(errors='replace')[-16000:]}",
                        pytrace=False,
                    )
        assert process.returncode == 0, f"Full log: {log_path}\n{log_path.read_text(errors='replace')[-16000:]}"

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
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="StreamFlow's local CWL execution uses POSIX shell commands; native Windows runs the CLI E2E instead",
)
@pytest.mark.parametrize("with_yaml", [False, True])
@pytest.mark.parametrize(
    "split_method,aggregate_method,refinement_method",
    [
        ("recursive", "auto", "none"),
        ("linear", "linear_belief_propagation", "conditioned"),
        ("recursive", "linear", "mean_field"),
        ("k_interactions", "recursive_graph", "soft_consensus"),
        ("recursive_graph", "linear_belief_propagation", "consensus"),
        ("quadtree", "quadtree", "soft_consensus"),
    ],
)
def test_actual_cwl_dataset_workflow_and_resume(
    tmp_path, run_command, with_yaml, split_method, aggregate_method, refinement_method
):
    pytest.importorskip("streamflow.main")
    matrices = {
        "first": np.triu(-np.ones((3, 3))),
        "second": np.diag([-2.0, -4.0, -6.0]),
    }
    records = [
        {
            "id": name,
            "dim": len(matrices[name]),
            "qubo_mat": [[i, j, float(matrices[name][i, j])] for i in range(3) for j in range(i, 3)],
        }
        for name in ["first", "second"]
    ]
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("\n".join(json.dumps(record) for record in records))
    common = tmp_path / "solver-common.yaml"
    private = tmp_path / "solver-private.yaml"
    common.write_text("QSPLIT_SOLVER_MODULE: invalid.module\n")
    private.write_text("QSPLIT_SOLVER_MODULE: qsplit.adapters.all_zero\nIQM_TOKEN: test-private-sentinel\n")
    refinement_solver = tmp_path / "refinement-solver.yaml"
    refinement_solver.write_text("QSPLIT_BACKEND: dwave\nIQM_TOKEN: refinement-private-sentinel\n")
    settings = tmp_path / "settings.json"
    store = tmp_path / "store"
    settings.write_text(
        json.dumps(
            {
                "dataset": {"class": "File", "path": str(dataset)},
                "cut_dim": 2,
                "split_method": split_method,
                "aggregate_method": aggregate_method,
                "refinement_method": refinement_method,
                "refinement_loops": 0 if refinement_method == "none" else 2,
                "refinement_solver_configs": [{"class": "File", "path": str(refinement_solver)}],
                "parallel_configs": [{"class": "File", "path": str(path)} for path in [common, private]]
                if with_yaml
                else [],
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
                "scheduling": {"scheduler": {"type": "default", "config": {"retry_delay": 1}}},
                "workflows": {
                    "local-test": {
                        "type": "cwl",
                        "config": {"file": str(REPO / "streamflow/cwl/main.cwl"), "settings": str(settings)},
                        "bindings": [
                            {"step": f"/qsplit_instances/{step}", "target": {"deployment": "cpu-simulator"}}
                            for step in ["parallelize", "refine/solve"]
                        ]
                        + [
                            {"step": f"/qsplit_instances/{step}", "target": {"deployment": "classical"}}
                            for step in [
                                "split",
                                "aggregate",
                                "initialize_refinement",
                                "refine/prepare",
                                "refine/update",
                            ]
                        ],
                    }
                },
                "deployments": {
                    "local": {"type": "local", "workdir": str(tmp_path / "work"), "config": {}},
                    "classical": {"type": "local", "workdir": str(tmp_path / "classical"), "config": {}},
                    "cpu-simulator": {
                        "type": "qsplit.quantum_connector",
                        "wraps": "local",
                        "config": {
                            "provider": "dwave",
                            "providerPool": ["dwave"],
                            "maxConcurrentJobs": {"dwave": 2},
                        },
                    },
                },
                "database": {"type": "sqlite", "config": {"connection": str(tmp_path / "streamflow.db")}},
            }
        )
    )
    run_command("streamflow", "run", "--debug", "--outdir", tmp_path / "output", config)
    assert len(list(store.glob("solutions_*.csv"))) == 2
    for path in store.glob("solutions_*.csv"):
        check_result(path, matrices[path.stem.removeprefix("solutions_")])
        assert "test-private-sentinel" not in path.read_text()
        assert "refinement-private-sentinel" not in path.read_text()
        if with_yaml and refinement_method == "none":
            frame = pd.read_csv(path)
            assert (frame[frame.backend == "dwave"].energy == 0).all()
    history_files = list((tmp_path / "output").rglob("refinement_history*.json"))
    assert len(history_files) == 2
    for history_file in history_files:
        history = json.loads(history_file.read_text())
        assert 1 <= len(history) <= 3
        if refinement_method == "none":
            assert len(history) == 1
        else:
            assert len(history) >= 2
            if with_yaml:
                assert history[0] == 0
                assert min(history) < history[0]
            if refinement_method == "conditioned":
                assert np.all(np.diff(history) <= 0)
    states = list((tmp_path / "output").rglob("state*.pkl"))
    assert len(states) == 2
    assert {load_qubo(path).instance_id for path in states} == set(matrices)
    for path in states:
        state = load_qubo(path)
        np.testing.assert_array_equal(state.mat[:3, :3], matrices[state.instance_id])
        assert state.solutions.energy.min() == min(state.refinement_history)
        assert b"test-private-sentinel" not in path.read_bytes()
        assert b"refinement-private-sentinel" not in path.read_bytes()
    original = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in store.glob("*.csv")}
    run_command("streamflow", "run", "--debug", "--outdir", tmp_path / "resumed", config)
    assert {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in store.glob("*.csv")} == original
    manifests = list((tmp_path / "resumed").rglob("dataset_results_manifest.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())["resolved_solutions"] == 2
