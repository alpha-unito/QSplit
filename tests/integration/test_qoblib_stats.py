import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("qoblib")
pytest.importorskip("pyscipopt")

from qoblib_stats import census, write_report


def test_only_valid_cli_filters_errors_and_sizes_without_printing_skips(tmp_path):
    source = tmp_path / "QOBLIB"
    binary = "Minimize\n obj: x\nSubject To\n c: x <= 1\nBounds\n 0 <= x <= 1\nBinary\n x\nEnd\n"
    files = {
        "01-marketsplit/instances/broken.dat": "broken",
        "02-labs/solutions/labs002.sol": "01",
        "02-labs/solutions/labs1000.sol": "0",
        "03-birkhoff/instances/collection.json": json.dumps(
            {
                "small": {"id": "small", "n": 2, "scale": 1, "scaled_doubly_stochastic_matrix": [1, 0, 0, 1]},
                "large": {"id": "large", "n": 10, "scale": 10000},
                "broken": {"id": "broken"},
                "another": {"id": "another", "n": 2, "scale": 1, "scaled_doubly_stochastic_matrix": [1, 0, 0, 1]},
            }
        ),
        "04-steiner/instances/missing/arcs.dat": "1 2 1",
        "05-sports/models/mixed_integer_linear/continuous.lp": binary.replace("Binary\n x\n", ""),
        "08-network/models/integer_lp/broken.lp": "Minimize\n obj: x\nSubject To\n c: x + <= 1\nEnd\n",
        "09-routing/models/integer_linear/tiny.lp": binary,
        "09-routing/instances/tiny.vrp": (
            "NAME: tiny\nDIMENSION: 2\nCAPACITY: 1\nNODE_COORD_SECTION\n1 0 0\n2 1 0\n"
            "DEMAND_SECTION\n1 0\n2 1\nDEPOT_SECTION\n1\n-1\nEOF\n"
        ),
        "09-routing/models/integer_linear/missing_source.lp": binary,
        "10-topology/models/seidel_linear/unbounded.lp": (
            "Minimize\n obj: x\nSubject To\n c: x >= 0\nGeneral\n x\nEnd\n"
        ),
    }
    for relative, text in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    output = tmp_path / "valid"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[2] / "qoblib_stats.py"),
            "--repository",
            str(source),
            "--output-dir",
            str(output),
            "--only-valid",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == ""
    assert result.stdout.rstrip().endswith("Skipped instances: 9")
    rows = list(csv.DictReader((output / "instances.csv").open()))
    assert len(rows) == 4
    assert {r["instance"] for r in rows} == {"2", "small", "another", "tiny"}
    assert all(not r["error"] and 1 <= int(r["qubo_variables"]) <= 100000 for r in rows)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["only_valid"] and summary["catalog_count"] == 4
    assert summary["statistics"]["qubo_variables"]["count"] == 4
    report = (output / "report.html").read_text()
    assert "broken" not in report
    assert '<html lang="en">' in report
    assert "Formulation variables" in report and "Histogram" in report


def test_only_valid_respects_matrix_budget_and_reports_empty_selection(tmp_path):
    path = tmp_path / "QOBLIB/02-labs/solutions/labs002.sol"
    path.parent.mkdir(parents=True)
    path.write_text("01")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    limits = {"max_variables": 100000, "max_matrix_bytes": 8}
    rows, skipped = census(path.parents[2], scratch, limits=limits)
    assert rows == [] and skipped == 1
    output = tmp_path / "empty"
    write_report(rows, output, path.parents[2], only_valid=True)
    assert list(csv.DictReader((output / "instances.csv").open())) == []
    assert json.loads((output / "summary.json").read_text())["statistics"]["variables"] == {"count": 0}
    assert "QOBLIB: 0 instances" in (output / "report.html").read_text()
    full, skipped = census(path.parents[2], scratch)
    assert len(full) == 1 and full[0]["qubo_variables"] == 3 and skipped == 0
