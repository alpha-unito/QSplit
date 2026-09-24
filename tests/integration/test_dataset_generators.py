import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.dataset


def test_maxcut_generation_uses_local_fixture_without_downloading(tmp_path, monkeypatch):
    pytest.importorskip("pennylane")
    generator = importlib.import_module("dataset.generate_max_cut_dataset")
    monkeypatch.chdir(tmp_path)
    dataset = SimpleNamespace(
        ns=[3, 3, 2, 101],
        ids=["triangle", "duplicate", "edge", "large"],
        edges=[[(0, 1), (1, 2), (0, 2), (0, 0)], [(0, 1)], [(0, 1)], []],
    )
    load = Mock(return_value=[dataset])
    monkeypatch.setattr(generator.qml.data, "load", load)
    generator.main()
    records = [json.loads(line) for line in (tmp_path / "qubo_max_cut.jsonl").read_text().splitlines()]
    assert [record["dim"] for record in records] == [2, 3]
    assert records[0]["qubo_mat"] == [[0, 0, -1], [1, 1, -1], [0, 1, 2]]
    assert records[1]["min"] == -2
    assert all(record["min"] <= record["sa"] <= record["max"] == 0 for record in records)
    assert all(0 <= record["sparsity"] <= 1 for record in records)
    load.assert_called_once()


def test_knapsack_generation_from_tiny_local_dataset(tmp_path, monkeypatch):
    pytest.importorskip("tqdm")
    generator = importlib.import_module("dataset.generate_kp_dataset")
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset/qubo_max_cut.jsonl").write_text('{"dim": 2}\n')
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    generator.main()
    record = json.loads((work / "qubo_kp.jsonl").read_text())
    assert record["dim"] >= 2
    assert record["qubo_mat"]
    assert all(0 <= row <= col < record["dim"] for row, col, _ in record["qubo_mat"])
    assert record["min"] <= record["sa"] <= record["max"]
