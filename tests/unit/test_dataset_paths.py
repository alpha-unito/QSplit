from pathlib import Path, PurePosixPath, PureWindowsPath
from unittest.mock import Mock

import pytest

from qsplit.cwl.cli import collect_dataset_results, dataset_prepare, persist_instance_solution

MODULES = [dataset_prepare, collect_dataset_results, persist_instance_solution]


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
@pytest.mark.parametrize("path_type", [PurePosixPath, PureWindowsPath])
def test_ephemeral_directory_detection_across_path_separators(module, path_type):
    path = Mock(spec=Path)
    path.resolve.return_value = path_type("/tmp/streamflow/run/solutions")
    assert module._is_ephemeral_solutions_dir(path)
    path.resolve.return_value = path_type("/project/solutions")
    assert not module._is_ephemeral_solutions_dir(path)


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
@pytest.mark.parametrize("source", ["absolute", "launch", "repo", "fallback", "cwd"])
def test_solution_directory_resolution(module, source, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("QSPLIT_LAUNCH_DIR", "QSPLIT_PROJECT_ROOT", "PWD", "OLDPWD", "INIT_CWD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path / "absent")
    expected = tmp_path / "solutions"
    raw = "solutions"
    if source == "absolute":
        raw = str(expected)
    elif source in {"launch", "repo"}:
        project = tmp_path / "project"
        (project / "qsplit").mkdir(parents=True)
        (project / "streamflow").mkdir()
        expected = project / "solutions"
        if source == "launch":
            monkeypatch.setenv("QSPLIT_PROJECT_ROOT", str(project))
        else:
            monkeypatch.setattr(module, "_repo_root", lambda: project)
    elif source == "fallback":
        launch = tmp_path / "launch"
        launch.mkdir()
        monkeypatch.setenv("INIT_CWD", str(launch))
        expected = launch / "solutions"
    assert module._resolve_solutions_dir(raw) == expected
    assert module._is_ephemeral_solutions_dir(Path("/tmp/streamflow/run/solutions"))
    assert not module._is_ephemeral_solutions_dir(expected)


@pytest.mark.parametrize("module", [dataset_prepare, collect_dataset_results], ids=lambda m: m.__name__)
@pytest.mark.parametrize(
    "content,valid",
    [
        (None, False),
        ("", False),
        ("a,b,c,d\n", False),
        ("a,b,c,d\nx,y,z,invalid\n", False),
        ("a,b\nx,y\n", False),
        ("node_id,backend,bitstring,energy\nroot,dwave,01,-2.5\n", True),
    ],
)
def test_solution_file_validation(module, content, valid, tmp_path):
    path = tmp_path / "result.csv"
    if content is not None:
        path.write_text(content)
    assert module._is_valid_solution_file(path) is valid


@pytest.mark.parametrize(
    "record,line,expected",
    [({"id": " A "}, 2, "A"), ({"originale_index": 7}, 2, "7"), ({}, 3, "3"), ({"id": " "}, 4, "4")],
)
def test_dataset_identifiers(record, line, expected):
    assert dataset_prepare._record_id(record, line) == expected
    assert dataset_prepare._safe_id("../..", 3) == "row_3"
