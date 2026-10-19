import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("qoblib")
pytest.importorskip("pyscipopt")

from qsplit.benchmark.bridge import DEFAULT_FORMULATION, finalize, prepare, read_request
from qsplit.benchmark.sources import catalog_entries


def test_catalog_selects_all_classes_and_portfolio_variants(tmp_path):
    paths = [
        "01-marketsplit/instances/ms.dat",
        "01-marketsplit/instances/ms.opt.sol",
        "01-marketsplit/instances/README.md",
        "02-labs/models/integer/lp_files/labs004.lp.xz",
        "02-labs/solutions/labs005.opt.sol",
        "03-birkhoff/instances/qbench_03_dense.json",
        "04-steiner/instances/stp_tiny/arcs.dat",
        "04-steiner/instances/stp_missing/arcs.dat",
        "04-steiner/models/integer_linear/lp_files/stp_tiny.lp.xz",
        "05-sports/instances/ITC2021/tiny.xml.gz",
        "05-sports/models/mixed_integer_linear/lp_files/ITC2021/tiny.lp.xz",
        "06-portfolio/instances/po_a010_t10_s00/stock_prices.txt.gz",
        "06-portfolio/instances/po_missing/stock_prices.txt",
        "06-portfolio/models/binary_quadratic_programming/lp_files/bqp_a010_t10_s00_b004_l0.lp.xz",
        "06-portfolio/models/binary_quadratic_programming/lp_files/bqp_a010_t10_s00_b004_l1e-3.lp.xz",
        "06-portfolio/models/another_formulation/lp_files/ignored.lp.xz",
        "06-portfolio/solutions/po_a010_t10_s00_b004_l1e-3.opt.sol",
        "07-independentset/instances/farm.gph",
        "08-network/instances/demand.txt",
        "08-network/models/integer_lp/lp_files/network05.lp.xz",
        "08-network/solutions/network06.opt.sol",
        "09-routing/instances/XSH-n20-k4-01.vrp",
        "09-routing/models/integer_linear/lp_files/XSH-n20-k4-01.lp.xz",
        "10-topology/instances/topology_1000000_16.dat",
        "10-topology/models/seidel_linear/lp_files/topology_15_3.lp.xz",
        "10-topology/solutions/topology_20_3.opt.gph",
    ]
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    entries = catalog_entries(tmp_path, DEFAULT_FORMULATION)
    assert len({e["problem"] for e in entries}) == 10
    assert [e["instance"] for e in entries if e["problem"] == "labs"] == ["4", "5"]
    markets = [e for e in entries if e["problem"] == "marketsplit"]
    assert len(markets) == 1 and markets[0]["instance"] == "ms"
    assert markets[0]["source_instance"] == "01-marketsplit/instances/ms.dat"
    portfolio = [e for e in entries if e["problem"] == "portfolio" and "model" in e]
    assert [e["parameters"] for e in portfolio] == [{"budget": 4, "lambda": 0}, {"budget": 4, "lambda": 0.001}]
    assert portfolio[1]["reference_solution"].endswith("l1e-3.opt.sol")
    assert {e["instance"] for e in entries if e["problem"] == "topology"} == {
        "topology_15_3",
        "topology_20_3",
        "topology_1000000_16",
    }
    assert {e["instance"] for e in entries if "model" not in e} >= {"stp_missing", "po_missing", "network06"}


def test_all_selection_reports_failures_and_preserves_strict_mode(tmp_path, capsys):
    source = tmp_path / "QOBLIB"
    files = {
        "01-marketsplit/instances/tiny.dat": "1 2\n1 1 1\n",
        "02-labs/solutions/labs004.opt.sol": "0100\n",
        "02-labs/solutions/labs100.opt.sol": "0\n",
        "03-birkhoff/instances/broken.json": "{}",
        "04-steiner/instances/missing/arcs.dat": "1 2 1\n",
        "04-steiner/instances/missing/terms.dat": "1 1\n2 1\n",
    }
    for relative, text in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    request = tmp_path / "request.yaml"
    data = {"instances": "all", "runs": 1, "check": "off", "max_variables": 16, "on_conversion_error": "skip"}
    request.write_text(yaml.safe_dump(data))
    dataset = prepare(request, tmp_path / "bundle", repository=source)
    progress = capsys.readouterr().err
    assert "conversion failed" in progress and "labs/100" in progress and "steiner/missing" in progress
    assert "writing dataset: 2 runs; conversion failures=3" in progress
    records = [json.loads(line) for line in dataset.read_text().splitlines()]
    assert {r["problem"] for r in records} == {"marketsplit", "labs"}
    manifest = json.loads((dataset.parent / "manifest.json").read_text())
    assert manifest["catalog_count"] == 5
    assert {e["problem"] for e in manifest["conversion_errors"]} == {"labs", "birkhoff", "steiner"}
    store = tmp_path / "solutions"
    store.mkdir()
    for record in records:
        with (store / f"solutions_{record['id']}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["node_id", "backend", "bitstring", "energy"])
            writer.writerow(["root", "aggregate", "0" * record["dim"], record["offset"]])
    report = json.loads(finalize(dataset.parent, store, tmp_path / "results").read_text())
    assert report["conversion_errors"] == manifest["conversion_errors"]
    assert report["catalog_count"] == 5
    with pytest.raises(ValueError, match="requires --repository"):
        prepare(request, tmp_path / "no_repository")
    data["on_conversion_error"] = "error"
    request.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="No Birkhoff sub-instance"):
        prepare(request, tmp_path / "strict", repository=source)
    data["on_conversion_error"] = "unknown"
    request.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="on_conversion_error"):
        read_request(request)
    with pytest.raises(ValueError, match="No QOBLIB instances"):
        catalog_entries(tmp_path / "solutions", DEFAULT_FORMULATION)
    with pytest.raises(ValueError, match="checkout not found"):
        catalog_entries(tmp_path / "absent", DEFAULT_FORMULATION)
    data.update(instances=[{"problem": "labs", "instance": "100"}], on_conversion_error="skip")
    request.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="No convertible instances"):
        prepare(request, tmp_path / "empty", repository=source)
    empty = json.loads((tmp_path / "empty/manifest.json").read_text())
    assert empty["count"] == 0 and len(empty["conversion_errors"]) == 1


def test_catalog_defaults_skip_unsupported_entries_and_budget_before_conversion(tmp_path, monkeypatch):
    source = tmp_path / "QOBLIB"
    files = {
        "02-labs/solutions/labs002.sol": "01\n",
        "02-labs/solutions/labs004.sol": "0100\n",
        "03-birkhoff/instances/large.json": json.dumps({"one": {"id": "one", "n": 100, "scale": 1000}}),
    }
    for relative, text in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    request = tmp_path / "request.yaml"
    request.write_text('instances: all\nruns: 1\ncheck: "off"\nmax_matrix_bytes: 72\n')
    parsed = read_request(request)
    assert parsed["on_conversion_error"] == "skip"
    assert parsed["max_variables"] == 100_000
    encoded_lengths = []
    from qsplit.benchmark.conversion import labs_record

    def encode_labs(n, **options):
        encoded_lengths.append(n)
        assert n == 2
        return labs_record(n, **options)

    monkeypatch.setattr("qsplit.benchmark.bridge.labs_record", encode_labs)
    dataset = prepare(request, tmp_path / "bundle", repository=source)
    manifest = json.loads((dataset.parent / "manifest.json").read_text())
    assert manifest["catalog_count"] == 3 and manifest["count"] == 1
    assert len(manifest["conversion_errors"]) == 2
    assert {e["problem"] for e in manifest["conversion_errors"]} == {"labs", "birkhoff"}
    assert encoded_lengths == [2]


def test_local_cli_reads_pipeline_yaml_and_reports_progress(tmp_path):
    pytest.importorskip("dwave.samplers")
    source = tmp_path / "source"
    source.mkdir()
    request = tmp_path / "request.yaml"
    request.write_text('runs: 1\ncheck: "off"\ninstances:\n- problem: labs\n  instance: "3"\n')
    config = tmp_path / "config.yaml"
    config.write_text(
        "CUT_DIM: 2\nSPLIT_METHOD: linear\nAGGREGATE_METHOD: linear_belief_propagation\n"
        "REFINEMENT_LOOPS: 1\nREFINEMENT_METHOD: consensus\n"
    )
    output = tmp_path / "output"
    command = [
        sys.executable,
        "-m",
        "qsplit.benchmark.local_benchmark",
        "--request",
        str(request),
        "--repository",
        str(source),
        "--output-dir",
        str(output),
        "--config",
        str(config),
    ]
    process = subprocess.run(
        command, cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=30
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads((output / "run.json").read_text())["configuration"]["CUT_DIM"] == 2
    assert json.loads((output / "results/provenance.json").read_text())["pipeline"]["cut_dim"] == 2
    assert json.loads((output / "run.json").read_text())["split_method"] == "linear"
    assert json.loads((output / "run.json").read_text())["aggregate_method"] == "linear_belief_propagation"
    for stage in ["setup", "prepare", "solve", "split", "aggregate", "refine", "export"]:
        assert f"QSPLIT BENCHMARK {stage}" in process.stderr
    assert "labs/labs003" in process.stderr and "1/1 remaining=0" in process.stderr
    assert "run=1" in process.stderr and "completed in" in process.stderr
    config.write_text("CUT_DIM: 0\n")
    process = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert process.returncode != 0 and "CUT_DIM must be a positive integer" in process.stderr
    process = subprocess.run(command + ["--cut-dim", "2"], capture_output=True, text=True, timeout=30)
    assert process.returncode != 0 and "unrecognized arguments" in process.stderr


def test_problem_class_selection_expands_only_requested_classes(tmp_path):
    source = tmp_path / "QOBLIB"
    files = {
        "01-marketsplit/instances/ignored.dat": "1 2\n1 1 1\n",
        "02-labs/solutions/labs004.opt.sol": "0100\n",
        "03-birkhoff/instances/ignored.json": "invalid JSON that must not be loaded",
        "07-independentset/instances/tiny.gph": "p edge 3 2\ne 1 2\ne 2 3\n",
    }
    for relative, text in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    request = tmp_path / "request.yaml"
    request.write_text('instances: [labs, independentset]\nruns: 2\ncheck: "off"\n')
    dataset = prepare(request, tmp_path / "bundle", repository=source)
    records = [json.loads(line) for line in dataset.read_text().splitlines()]
    assert len(records) == 4
    assert {(r["problem"], r["instance"]) for r in records} == {("labs", "labs004"), ("independentset", "tiny")}
    manifest = json.loads((dataset.parent / "manifest.json").read_text())
    assert manifest["catalog_count"] == 2 and not manifest["conversion_errors"]
    with pytest.raises(ValueError, match="requires --repository"):
        prepare(request, tmp_path / "no_checkout")


@pytest.mark.parametrize("selection", [["labs", "indipendentset"], [{"problem": "indipendentset", "instance": "tiny"}]])
def test_unknown_problem_class_has_actionable_error(tmp_path, selection):
    request = tmp_path / "request.yaml"
    request.write_text(yaml.safe_dump({"instances": selection}))
    with pytest.raises(ValueError, match="did you mean 'independentset'"):
        read_request(request)


@pytest.mark.parametrize("selection", [["labs", {"problem": "labs", "instance": "4"}], [1], [{"problem": "labs"}]])
def test_malformed_selection_does_not_raise_type_error(tmp_path, selection):
    request = tmp_path / "request.yaml"
    request.write_text(yaml.safe_dump({"instances": selection}))
    with pytest.raises(ValueError, match="instances must be"):
        read_request(request)
