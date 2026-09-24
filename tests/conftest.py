"""Local-only test defaults and shared, independent correctness oracles."""

import os
import socket
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qsplit.qubo import QUBO


@pytest.fixture(autouse=True)
def local_environment(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith(("QSPLIT_", "IQM_", "REFINEMENT_", "QUANTUM_")) or name in {
            "TOKEN_IBM",
            "CRN_IBM",
            "DWAVE_API_TOKEN",
            "CUT_DIM",
            "EXACT_RATIO",
        }:
            monkeypatch.delenv(name)
    monkeypatch.setenv("CUT_DIM", "2")
    monkeypatch.setenv("QSPLIT_BACKEND", "dwave")
    monkeypatch.setenv("QSPLIT_IQM_STATE_DIR", str(tmp_path / "iqm_state"))
    monkeypatch.setenv("QSPLIT_LAUNCH_DIR", str(tmp_path))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    # Never let an accidental provider call use the network. Local IPC is needed
    # by simulators and the local workflow engine.
    connect = socket.socket.connect

    def local_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            assert address[0] in {"127.0.0.1", "::1", "localhost"}, f"Non-local network access: {address}"
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_connect)


def pytest_collection_modifyitems(items):
    for item in items:
        parts = Path(item.path).parts
        level = "e2e" if "e2e" in parts else "integration" if "integration" in parts else "unit"
        item.add_marker(getattr(pytest.mark, level))


@pytest.fixture
def make_qubo():
    def make(matrix, ids=None, offset=0):
        matrix = np.array(matrix, dtype=float)
        ids = np.arange(len(matrix)) if ids is None else np.array(ids)
        return QUBO(matrix, ids.copy(), ids.copy(), offset=offset)

    return make


@pytest.fixture
def exact_solver():
    """Enumerate the original binary objective, independently of any adapter."""

    def solve(qubo):
        ids = sorted((set(qubo.rows_idx) | set(qubo.cols_idx)) - {-1})
        samples = []
        for bits in product((0, 1), repeat=len(ids)):
            assignment = dict(zip(ids, bits))
            rows = np.array([assignment.get(idx, 0) for idx in qubo.rows_idx])
            cols = np.array([assignment.get(idx, 0) for idx in qubo.cols_idx])
            samples.append({**assignment, "energy": float(rows @ qubo.mat @ cols + qubo.offset)})
        df = pd.DataFrame(samples)
        return df[df.energy == df.energy.min()].reset_index(drop=True)

    return solve


@pytest.fixture
def assert_solution():
    def check(qubo, solutions=None):
        df = qubo.solutions if solutions is None else solutions
        ids = {idx for idx in set(qubo.rows_idx) | set(qubo.cols_idx) if idx >= 0}
        assert df is not None and not df.empty
        assert set(df.columns) == ids | {"energy"}
        assert np.isfinite(df.to_numpy()).all()
        assert df[list(ids)].isin([0, 1]).all().all()
        for _, sample in df.iterrows():
            rows = np.array([sample[idx] if idx >= 0 else 0 for idx in qubo.rows_idx])
            cols = np.array([sample[idx] if idx >= 0 else 0 for idx in qubo.cols_idx])
            assert sample.energy == pytest.approx(rows @ qubo.mat @ cols + qubo.offset)

    return check
