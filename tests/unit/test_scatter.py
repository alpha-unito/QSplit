import sys
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from qsplit.cwl.cli import scatter
from qsplit.cwl.cli.utils import load_qubo, save_qubo


@pytest.mark.parametrize(
    "raw,expected",
    [("", "dwave"), ("any", "dwave"), (" IBM ", "ibm"), ("dummy-solver", "dummy"), ("dummy_solver", "dummy")],
)
def test_backend_resolution(raw, expected):
    assert scatter.resolve_backend(raw) == expected


def test_backend_override(monkeypatch):
    monkeypatch.setenv("QSPLIT_BACKEND", "ibm")
    assert scatter.resolve_backend("auto") == "ibm"
    monkeypatch.setenv("QSPLIT_SOLVER_MODULE", "qsplit.adapters.all_zero")
    assert scatter.load_solver("ibm").__module__ == "qsplit.adapters.all_zero"
    monkeypatch.setenv("QSPLIT_SOLVER_MODULE", "qsplit.qubo")
    with pytest.raises(AttributeError, match="not callable"):
        scatter.load_solver("ibm")


@pytest.mark.parametrize(
    "result,error,match",
    [
        (None, RuntimeError, "no result"),
        ([], TypeError, "DataFrame"),
        (pd.DataFrame(), RuntimeError, "empty"),
        (pd.DataFrame({0: [1]}), RuntimeError, "energy"),
        (pd.DataFrame({0: [1], "energy": [np.nan]}), RuntimeError, "NaN"),
    ],
)
def test_rejects_invalid_solver_results(result, error, match, tmp_path, monkeypatch, make_qubo):
    source, output = tmp_path / "input.pkl", tmp_path / "output.pkl"
    save_qubo(source, make_qubo([[-1]]))
    monkeypatch.setattr(sys, "argv", ["scatter", "--input-qubo", str(source), "--output-qubo", str(output)])
    monkeypatch.setattr(scatter, "load_solver", lambda _: lambda qubo: result)
    with pytest.raises(error, match=match):
        scatter.main()
    assert not output.exists()


def test_rejects_non_qubo_input(tmp_path, monkeypatch):
    path = tmp_path / "wrong.pkl"
    save_qubo(path, {"not": "qubo"})
    monkeypatch.setattr(sys, "argv", ["scatter", "--input-qubo", str(path), "--output-qubo", str(tmp_path / "out")])
    with pytest.raises(TypeError, match="Expected QUBO"):
        scatter.main()


@pytest.mark.parametrize("change", ["matrix", "offset", "rows", "cols", "shape", "empty", "energy", "nan"])
def test_cache_invalidates_incompatible_or_incomplete_results(change, tmp_path, make_qubo):
    qubo = make_qubo(-np.eye(2), offset=2)
    cached = deepcopy(qubo)
    cached.solutions = pd.DataFrame({0: [1], 1: [1], "energy": [0.0]})
    if change == "matrix":
        cached.mat[0, 0] += 1
    elif change == "offset":
        cached.offset += 1
    elif change == "rows":
        cached.rows_idx = cached.rows_idx[::-1]
    elif change == "cols":
        cached.cols_idx = cached.cols_idx[::-1]
    elif change == "shape":
        cached = make_qubo([[-1]])
    elif change == "empty":
        cached.solutions = pd.DataFrame()
    elif change == "energy":
        cached.solutions = cached.solutions.drop(columns="energy")
    elif change == "nan":
        cached.solutions["energy"] = np.nan
    path = tmp_path / "cached.pkl"
    save_qubo(path, cached)
    assert scatter._load_cached_iqm_dataframe(path, qubo) is None


def test_iqm_cache_store_hit_and_probe_miss_without_provider(tmp_path, monkeypatch, make_qubo):
    qubo = make_qubo([[-1]])
    qubo.instance_id, qubo.node_id = "case", "root"
    source, output = tmp_path / "input.pkl", tmp_path / "output.pkl"
    save_qubo(source, qubo)
    monkeypatch.setenv("QSPLIT_BACKEND", "iqm")
    monkeypatch.setattr(scatter, "_resolve_iqm_subproblem_dir", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["scatter", "--input-qubo", str(source), "--output-qubo", str(output)])
    fake_solver = Mock(return_value=pd.DataFrame({0: [1], "energy": [-1.0]}))
    loader = Mock(return_value=fake_solver)
    monkeypatch.setattr(scatter, "load_solver", loader)
    scatter.main()
    fake_solver.assert_called_once()
    assert len(list((tmp_path / "case").glob("*.pkl"))) == 1
    assert not list(tmp_path.rglob("*.tmp"))
    output.unlink()
    loader.reset_mock()
    monkeypatch.setenv("QSPLIT_IQM_CACHE_ONLY", "true")
    scatter.main()
    loader.assert_not_called()
    assert load_qubo(output).solutions.iloc[0].energy == -1
    qubo.offset = 99
    save_qubo(source, qubo)
    output.unlink()
    with pytest.raises(SystemExit) as exc:
        scatter.main()
    assert exc.value.code == 75
    loader.assert_not_called()
    assert not output.exists()


def test_cache_coordinates_sanitize_paths(monkeypatch):
    monkeypatch.setenv("IQM_QUANTUM_COMPUTER", "machine / one")
    monkeypatch.setenv("QUANTUM_TUNE_QAOA", "yes")
    coords = scatter._iqm_cache_coordinates(SimpleNamespace(instance_id="../../escape"), "root.pkl")
    assert coords == ("escape", "root", "machine_one", "yes")
