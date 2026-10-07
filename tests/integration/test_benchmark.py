import csv
import itertools
import json
import os
import subprocess
import sysconfig
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

pytest.importorskip("qoblib")
pytest.importorskip("pyscipopt")

from qsplit.benchmark.bridge import finalize, prepare, serialize, solution_object
from qsplit.benchmark.conversion import decode, encode_model, labs_record, model_evaluation, read_lp, to_qubo
from qsplit.benchmark.local_benchmark import run

REPO = Path(__file__).resolve().parents[2]


def energy(record, bits):
    return record["offset"] + sum(c * bits[i] * bits[j] for i, j, c in record["qubo_mat"])


def test_integer_constraint_encoding_is_exact_including_slack_offset_and_maximization():
    model = {
        "variables": {"x": [-1, 2], "y": [0, 1]},
        "sense": "maximize",
        "constant": 4,
        "linear": {"x": 2, "y": -1},
        "quadratic": [["x", "y", 3]],
        "constraints": [{"coefficients": {"x": 0.5, "y": 0.5}, "lower": 0.5, "upper": 1}],
    }
    record = encode_model(model)
    minimum = {}
    for bits in itertools.product([0, 1], repeat=record["dim"]):
        values = decode(record, "".join(map(str, bits)))
        key = tuple(values.items())
        minimum[key] = min(minimum.get(key, np.inf), energy(record, bits))
    for key, e in minimum.items():
        values = dict(key)
        objective, violations = model_evaluation(model, values)
        if not violations:
            assert e == pytest.approx(-objective)
        else:
            assert e >= -objective + record["bridge"]["penalty"]
    matrix = to_qubo(record)
    bits = np.ones(record["dim"])
    assert bits @ matrix.mat @ bits + matrix.offset == pytest.approx(energy(record, bits))


@pytest.mark.parametrize("n", [1, 3, 4, 5])
def test_labs_quadratization_matches_original_energy_for_every_sequence(n):
    record = labs_record(n)
    for bits in itertools.product([0, 1], repeat=n):
        x = list(bits) + [bits[i] * bits[j] for i in range(n) for j in range(i + 1, n)]
        spin = [2 * b - 1 for b in bits]
        original = sum(sum(spin[i] * spin[i + k] for i in range(n - k)) ** 2 for k in range(1, n))
        assert energy(record, x) == original
    if n == 3:
        for x in itertools.product([0, 1], repeat=record["dim"]):
            spin = [2 * b - 1 for b in x[:n]]
            original = sum(sum(spin[i] * spin[i + k] for i in range(n - k)) ** 2 for k in range(1, n))
            assert energy(record, x) >= original


def test_lp_recovers_quadratic_objective_without_the_continuous_epigraph(tmp_path):
    lp = tmp_path / "quadratic.lp"
    lp.write_text(
        "Minimize\n obj: 2 x + [ 6 x * y + 4 y^2 ] / 2\nSubject To\n c: x + y <= 1\n"
        "Bounds\n 0 <= x <= 1\n 0 <= y <= 1\nBinary\n x y\nEnd\n"
    )
    model = read_lp(lp)
    assert set(model["variables"]) == {"x", "y"}
    assert model_evaluation(model, {"x": 1, "y": 1})[0] == 7
    record = encode_model(model)
    assert "quadobjvar" not in record["bridge"]["mapping"]


@pytest.mark.parametrize("bits", ["", "0", "02", "000"])
def test_decode_rejects_incomplete_or_nonbinary_samples(bits):
    record = encode_model({"variables": {"x": [0, 1], "y": [0, 1]}})
    with pytest.raises(ValueError, match="complete binary"):
        decode(record, bits)


def test_conversion_rejects_unbounded_fractional_or_oversized_domains():
    for bounds in [[0, float("inf")], [0, 1.5], [2, 1]]:
        with pytest.raises(ValueError):
            encode_model({"variables": {"x": bounds}})
    with pytest.raises(ValueError, match="max_variables"):
        labs_record(5, max_variables=5)


@pytest.fixture
def source_request(tmp_path):
    source = tmp_path / "QOBLIB"
    data = {
        "01-marketsplit/instances/tiny.dat": "1 2\n1 1 1\n",
        "02-labs/solutions/labs004.opt.sol": "# reference energy 2\n0100\n",
        "07-independentset/instances/tiny.gph": "p edge 3 2\ne 1 2\ne 2 3\n",
        "03-birkhoff/instances/tiny.json": json.dumps(
            {"one": {"id": "one", "n": 2, "scale": 1, "scaled_doubly_stochastic_matrix": [1, 0, 0, 1]}}
        ),
    }
    for name, text in data.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    request = tmp_path / "request.yaml"
    request.write_text(
        yaml.safe_dump(
            {
                "runs": 2,
                "check": "off",
                "instances": [
                    {"problem": slug, "instance": name}
                    for slug, name in [
                        ("marketsplit", "tiny"),
                        ("independentset", "tiny"),
                        ("labs", "4"),
                        ("birkhoff", "tiny"),
                    ]
                ],
            }
        )
    )
    return source, request


@pytest.mark.parametrize("method", ["recursive", "linear", "recursive_graph", "quadtree"])
def test_local_pipeline_creates_portable_native_solutions_and_honest_drafts(
    tmp_path, source_request, method, monkeypatch
):
    pytest.importorskip("dwave.samplers")
    from qsplit.adapters.dwave import dwave_sa

    solver = Mock(wraps=dwave_sa.solve)
    monkeypatch.setattr(dwave_sa, "solve", solver)
    source, request = source_request
    result = run(
        request,
        tmp_path / method,
        repository=source,
        split_method=method,
        config={"CUT_DIM": 4, "REFINEMENT_LOOPS": 1, "QSPLIT_SOLVER_MODULE": "invalid.module"},
    )
    assert solver.call_count > 0
    settings = json.loads((result.parent.parent / "run.json").read_text())["configuration"]
    assert settings["QSPLIT_BACKEND"] == "dwave"
    assert "QSPLIT_SOLVER_MODULE" not in settings
    report = json.loads(result.read_text())
    assert len(report["runs"]) == 8
    assert all(r["solution_file"] and r["checker_status"] == "NOT_CHECKED" for r in report["runs"])
    assert all(r["feasible"] is None and not r["submission_ready"] for r in report["runs"])
    assert all(r["qubo_energy"] == pytest.approx(r["matrix_energy"] + r["offset"]) for r in report["runs"])
    assert all(s["feasible_runs"] == 0 for s in report["summary"])
    for path in result.parent.rglob("*_summary.csv"):
        with path.open() as f:
            row = next(csv.DictReader(f))
        assert row["# Runs"] == "2" and row["# Feasible Runs"] == "0" and row["Optimality Bound"] == "N/A"
    bundle = result.parent.parent / "bundle"
    finalize(bundle, result.parent.parent / "solutions", tmp_path / "exported_again")
    (bundle / "source/01-marketsplit/instances/tiny.dat").write_text("modified")
    with pytest.raises(ValueError, match="changed"):
        finalize(bundle, result.parent.parent / "solutions", tmp_path / "tampered")


def test_official_verdict_controls_feasibility_and_objective(monkeypatch, tmp_path, source_request):
    pytest.importorskip("dwave.samplers")
    import qoblib
    from qoblib._problem import Problem

    source, request = source_request
    data = yaml.safe_load(request.read_text())
    data["check"] = "required"
    data["instances"] = [{"problem": "labs", "instance": "4"}]
    request.write_text(yaml.safe_dump(data))

    def check(self, instance, solution):
        return qoblib.CheckResult(True, status="VALID", objective=self.compute_objective(instance, solution))

    monkeypatch.setattr(Problem, "check_solution", check)
    report = json.loads(run(request, tmp_path / "official", repository=source).read_text())
    assert all(r["feasible"] and r["submission_ready"] for r in report["runs"])
    assert report["summary"][0]["feasible_runs"] == 2
    assert not report["summary"][0]["submission_ready"]


def test_original_solution_formats_for_remaining_classes():
    base = {"instance": "tiny", "parameters": {}, "variable_aliases": {}, "bridge": {"model": {}}}
    cases = [
        ("steiner", SimpleNamespace(), {"y#1#2#1": 1}, "1 2 1"),
        ("network", SimpleNamespace(), {"z": 3, "x#1#2": 1, "f#1#1#2": 3}, "f#1#1#2 3"),
        ("routing", SimpleNamespace(depot=1, coords={1: (0, 0), 2: (1, 0)}), {"x#1#2": 1, "x#2#1": 1}, "Route #1: 2"),
        ("topology", SimpleNamespace(n=3, degree=2), {"dist#0#1#0": 1, "dist#1#2#0": 1}, "Diameter 2"),
        ("sports", SimpleNamespace(), {"x#0#1#2": 1}, 'home="0" away="1" slot="2"'),
    ]
    for slug, instance, values, expected in cases:
        solution, _ = solution_object({**base, "problem": slug}, instance, values)
        assert expected in serialize(solution, slug, instance)
    record = {
        **base,
        "problem": "portfolio",
        "parameters": {"budget": 4, "lambda": 0},
        "bridge": {"model": {"linear": {"x$AAPL#1#1#0": -2}}},
    }
    solution, _ = solution_object(record, SimpleNamespace(), {"x$AAPL#1#1#0": 1, "x$AAPL#1#_1#0": 0})
    assert "0 AAPL 1 0" in solution.to_checker_string()
    assert "objective -2" in solution.to_checker_string()


def test_export_rejects_leaf_only_and_incomplete_results(tmp_path, source_request):
    source, request = source_request
    bundle = tmp_path / "bundle"
    dataset = prepare(request, bundle, repository=source)
    records = [json.loads(line) for line in dataset.read_text().splitlines()]
    store = tmp_path / "store"
    store.mkdir()
    for record in records:
        (store / f"solutions_{record['id']}.csv").write_text("node_id,backend,bitstring,energy\nroot_0,dwave,0,-100\n")
    with pytest.raises(ValueError, match="complete root"):
        finalize(bundle, store, tmp_path / "results")


@pytest.mark.streamflow
def test_actual_streamflow_benchmark_bridges(tmp_path, source_request):
    pytest.importorskip("dwave.samplers")
    pytest.importorskip("streamflow.main")
    source, request = source_request
    request.write_text(
        yaml.safe_dump({"runs": 2, "check": "off", "instances": [{"problem": "marketsplit", "instance": "tiny"}]})
    )
    settings = tmp_path / "settings.yml"
    settings.write_text(
        yaml.safe_dump(
            {
                "benchmark_request": {"class": "File", "path": str(request)},
                "qoblib_repository": {"class": "Directory", "path": str(source)},
                "solutions_store_dir": str(tmp_path / "store"),
                "cut_dim": 2,
                "split_method": "recursive_graph",
                "refinement_method": "conditioned",
                "refinement_loops": 1,
            }
        )
    )
    config = tmp_path / "streamflow.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "version": "v1.0",
                "database": {"type": "sqlite", "config": {"connection": str(tmp_path / "streamflow.db")}},
                "workflows": {
                    "benchmark": {
                        "type": "cwl",
                        "config": {"file": str(REPO / "streamflow/cwl/benchmark.cwl"), "settings": str(settings)},
                    }
                },
            }
        )
    )
    env = {
        **os.environ,
        "PATH": sysconfig.get_path("scripts") + os.pathsep + os.environ["PATH"],
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    output = tmp_path / "output"
    log_path = tmp_path / "streamflow.log"
    with log_path.open("w") as log:
        process = subprocess.run(
            ["streamflow", "run", "--outdir", str(output), str(config)],
            cwd=tmp_path,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
    assert process.returncode == 0, log_path.read_text()[-12000:]
    reports = list(output.rglob("results.json"))
    assert reports
    report = json.loads(reports[0].read_text())
    assert len(report["runs"]) == 2
    assert all(r["solution_file"] and r["model_feasible"] for r in report["runs"])
    assert all(r["qubo_energy"] == 0 and r["offset"] > 0 for r in report["runs"])


def test_required_checker_errors_are_not_reported_as_feasible(monkeypatch, tmp_path, source_request):
    pytest.importorskip("dwave.samplers")
    import qoblib
    from qoblib._problem import Problem

    source, request = source_request
    data = {"runs": 1, "check": "required", "instances": [{"problem": "labs", "instance": "4"}]}
    request.write_text(yaml.safe_dump(data))

    def unavailable(self, instance, solution):
        raise qoblib.QoblibError("checker unavailable in test")

    monkeypatch.setattr(Problem, "check_solution", unavailable)
    with pytest.raises(qoblib.QoblibError, match="checker unavailable"):
        run(request, tmp_path / "required", repository=source)
    data["check"] = "auto"
    request.write_text(yaml.safe_dump(data))
    report = json.loads(run(request, tmp_path / "auto", repository=source).read_text())
    assert report["runs"][0]["checker_status"] == "UNAVAILABLE"
    assert report["runs"][0]["feasible"] is None
    assert report["runs"][0]["reference_objective"] == 2
    assert report["runs"][0]["improvement"] is None


def test_collection_expands_to_separate_birkhoff_submission_instances(tmp_path, source_request):
    source, request = source_request
    path = source / "03-birkhoff/instances/tiny.json"
    data = json.loads(path.read_text())
    data["two"] = {**data["one"], "id": "B2_4_2"}
    data["one"]["id"] = "B2_4_1"
    path.write_text(json.dumps(data))
    request.write_text(
        yaml.safe_dump({"runs": 1, "check": "off", "instances": [{"problem": "birkhoff", "instance": "tiny"}]})
    )
    dataset = prepare(request, tmp_path / "bundle", repository=source)
    records = [json.loads(line) for line in dataset.read_text().splitlines()]
    assert {r["instance"] for r in records} == {"B2_4_1", "B2_4_2"}
    assert {r["source_instance"] for r in records} == {"tiny"}
    assert all(len(r["bridge"]["model"]["permutations"]) == 1 for r in records)


def test_dataset_workflow_does_not_round_fractional_qubo_coefficients(tmp_path, monkeypatch):
    import sys

    from qsplit.cwl.cli.dataset_prepare import main

    coefficients = [0.123456789012345, -1.2345678912345]
    dataset = tmp_path / "fractional.jsonl"
    dataset.write_text(
        json.dumps({"id": "precise", "dim": 2, "qubo_mat": [[0, 0, coefficients[0]], [0, 1, coefficients[1]]]})
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["cli_dataset_prepare", "--dataset-jsonl", str(dataset), "--solutions-dir", str(tmp_path / "store")],
    )
    main()
    matrix = np.loadtxt(next((tmp_path / "dataset_matrices").glob("*.csv")), delimiter=",")
    assert matrix[0, :].tolist() == coefficients


def test_lp_selection_rejects_ambiguous_formulation_variants():
    from qsplit.benchmark.sources import load_model

    entries = [
        {"name": f"tiny_b{i}", "filename": f"tiny_b{i}.lp", "path": f"class/models/integer/tiny_b{i}.lp"}
        for i in [1, 2]
    ]
    problem = SimpleNamespace(files=lambda **kw: entries, load_model=lambda path, **kw: path)
    with pytest.raises(ValueError, match="exact model path"):
        load_model(problem, "tiny", "integer")
    assert load_model(problem, "tiny_b1", "integer") == entries[0]["path"]


def test_portfolio_variants_have_distinct_ids_and_keep_original_data_source(tmp_path):
    source = tmp_path / "source"
    instance = source / "06-portfolio/instances/po_tiny"
    instance.mkdir(parents=True)
    (instance / "stock_prices.txt").write_text("0 AAPL 1\n")
    (instance / "covariance_matrices.txt").write_text("0 AAPL AAPL 0\n")
    entries = []
    for budget in [1, 2]:
        model = source / f"06-portfolio/models/binary_quadratic_programming/bqp_tiny_b{budget}.lp"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_text(
            "Minimize\n obj: x$AAPL#1#1#0\nSubject To\n c: x$AAPL#1#1#0 <= 1\n"
            "Bounds\n 0 <= x$AAPL#1#1#0 <= 1\nGeneral\n x$AAPL#1#1#0\nEnd\n"
        )
        entries.append(
            {
                "problem": "portfolio",
                "instance": "po_tiny",
                "model": model.relative_to(source).as_posix(),
                "parameters": {"budget": budget, "lambda": 0},
            }
        )
    request = tmp_path / "request.yaml"
    request.write_text(yaml.safe_dump({"runs": 1, "check": "off", "instances": entries}))
    dataset = prepare(request, tmp_path / "bundle", repository=source)
    records = [json.loads(line) for line in dataset.read_text().splitlines()]
    assert {r["instance"] for r in records} == {"bqp_tiny_b1", "bqp_tiny_b2"}
    assert {r["source_instance"] for r in records} == {"po_tiny"}
    solution, _ = solution_object(records[0], SimpleNamespace(), {"x$AAPL#1#1#0": 1})
    assert solution.instance_name == "po_tiny"
