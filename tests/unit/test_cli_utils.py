from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qsplit.cwl.cli import dataset_prepare, persist_instance_solution, split, utils


@pytest.mark.parametrize(
    "raw,expected",
    [("", []), (" a.pkl, ,b.pkl ", ["a.pkl", "b.pkl"]), (["a.pkl,b.pkl", "", " c.pkl"], ["a.pkl", "b.pkl", "c.pkl"])],
)
def test_solved_paths(raw, expected):
    assert utils.parse_solved_paths(raw) == [Path(p) for p in expected]


def test_pickle_roundtrip_preserves_metadata(tmp_path, make_qubo):
    qubo = make_qubo([[1, -2], [0, 3]], ids=[20, 10], offset=7)
    qubo.node_id = "root_2"
    qubo.instance_id = "example"
    qubo.solutions = pd.DataFrame({20: [1], 10: [0], "energy": [8.0]})
    path = tmp_path / "node.pkl"
    utils.save_qubo(path, qubo)
    restored = utils.load_qubo(path)
    np.testing.assert_array_equal(restored.mat, qubo.mat)
    np.testing.assert_array_equal(restored.rows_idx, qubo.rows_idx)
    assert (restored.offset, restored.node_id, restored.instance_id) == (7, "root_2", "example")
    pd.testing.assert_frame_equal(restored.solutions, qubo.solutions)
    corrupt = tmp_path / "corrupt.pkl"
    corrupt.write_text("not a pickle")
    wrong = tmp_path / "wrong.pkl"
    utils.save_qubo(wrong, {"not": "qubo"})
    entries, by_id = utils.load_solved_qubos([corrupt, wrong, tmp_path / "missing", path])
    assert len(entries) == 1
    assert list(by_id) == ["root_2"]


@pytest.mark.parametrize(
    "row,expected",
    [
        (pd.Series({2: 1, 1: 0}), "01"),
        (pd.Series({"1": 1, "2": 0}), "10"),
        (pd.Series({1: np.nan, 2: "bad"}), ""),
        (pd.Series([1, 0, 0], index=[1, 1, 2]), "10"),
    ],
)
def test_bitstring_order_and_invalid_values(row, expected):
    assert utils.bitstring_from_row(row, [1, 2, 99]) == expected


@pytest.mark.parametrize("raw,expected", [("auto", None), (" AUTO ", None), ("", 1), ("0", 0), ("3", 3)])
def test_backend_job_counts(raw, expected):
    assert split._parse_count(raw) == expected


@pytest.mark.parametrize("raw", ["-1", "1.2", "invalid"])
def test_invalid_backend_counts(raw):
    with pytest.raises(ValueError, match="Invalid count"):
        split._parse_count(raw)


@pytest.mark.parametrize(
    "counts,expected",
    [(("1", "2", "0"), [1, 2, 0, 4]), (("auto", "auto", "auto"), [2, 2, 1, 2]), (("99", "2", "1"), [7, 0, 0, 0])],
)
def test_allocation_neither_loses_nor_duplicates_jobs(counts, expected):
    jobs = [Path(f"root_{i}.pkl") for i in range(7)]
    result = split._allocate_subproblems(
        jobs,
        enable_iqm=True,
        enable_quantinuum_h2=True,
        enable_quantinuum_h2e=True,
        iqm_real_jobs=counts[0],
        quantinuum_h2_real_jobs=counts[1],
        quantinuum_h2e_real_jobs=counts[2],
    )
    assert [len(v) for v in result.values()] == expected
    assert sorted(p for values in result.values() for p in values) == jobs
    assert len(jobs) == 7


def test_materialize_replaces_stale_files_and_falls_back_to_copy(tmp_path, monkeypatch):
    source = tmp_path / "root.pkl"
    source.write_bytes(b"content")
    output = tmp_path / "planned"
    output.mkdir()
    (output / "stale.pkl").touch()
    monkeypatch.setattr(split.os, "symlink", lambda *args: (_ for _ in ()).throw(OSError("unsupported")))
    split._materialize([source], output)
    assert [p.name for p in output.iterdir()] == ["000000_root.pkl"]
    assert next(output.iterdir()).read_bytes() == b"content"


@pytest.mark.parametrize(
    "name,expected",
    [("000000_000012_case A.csv", "case_A"), ("123_problem.csv", "123_problem"), ("---.csv", "instance_unknown")],
)
def test_instance_names(name, expected):
    assert split._instance_id_from_matrix_path(name) == expected
    assert persist_instance_solution._safe_id_from_matrix(Path(name)) == expected


@pytest.mark.parametrize("record", [{}, {"dim": 0}, {"dim": "bad"}, {"dim": None}])
def test_invalid_dataset_dimensions(record):
    assert dataset_prepare._matrix_from_record(record) is None


def test_sparse_dataset_matrix_normalizes_lower_triangle_and_ignores_bad_terms():
    record = {"dim": 2, "qubo_mat": [[0, 0, -1], [1, 0, 3], [2, 0, 9], [0], ["bad", 1, 2], None]}
    assert dataset_prepare._matrix_from_record(record) == [[-1.0, 3.0], [0.0, 0.0]]
