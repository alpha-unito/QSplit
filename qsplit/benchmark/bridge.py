import csv
import hashlib
import json
import platform
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from qsplit.benchmark.conversion import decode, encode_model, labs_record, model_evaluation, native_model, read_lp
from qsplit.benchmark.sources import DIRECTORIES, get_problem, load_instance, load_model, repository_ref, source_hashes

NATIVE = {"marketsplit", "independentset", "labs", "birkhoff"}
DEFAULT_FORMULATION = {
    "steiner": "integer_linear",
    "sports": "mixed_integer_linear",
    "portfolio": "binary_quadratic_programming",
    "network": "integer_lp",
    "routing": "integer_linear",
    "topology": "seidel_linear",
}
SUBMISSION_COLUMNS = (
    "Problem,Submitter,Affiliation,Date,Reference,Best Objective Value,Optimality Bound,Modeling Approach,"
    "# Decision Variables,# Binary Variables,# Integer Variables,# Continuous Variables,# Non-Zero Coefficients,"
    "Coefficients Type,Coefficients Range,Workflow,Algorithm Type,Paradigm,# Runs,# Feasible Runs,# Successful Runs,"
    "Success Threshold,Hardware Specifications,Total Runtime,Time to Solution,CPU Runtime,GPU Runtime,QPU Runtime,"
    "Other HW Runtime,Remarks"
).split(",")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_request(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("instances"), list) or not data["instances"]:
        raise ValueError("Benchmark request must contain a nonempty instances list")
    runs = data.get("runs", 5)
    maximum = data.get("max_variables", 4096)
    if data.get("check") is False:
        data["check"] = "off"
    if type(runs) is not int or runs < 1 or type(maximum) is not int or maximum < 1:
        raise ValueError("runs and max_variables must be positive integers")
    matrix_bytes = data.get("max_matrix_bytes", 256 * 1024**2)
    if type(matrix_bytes) is not int or matrix_bytes <= 0:
        raise ValueError("max_matrix_bytes must be a positive integer")
    if data.get("check", "auto") not in {"auto", "required", "off"}:
        raise ValueError("check must be auto, required or off")
    names = [str(e["instance"]) for e in data["instances"]] + [data.get("submission_name", "QSplit")]
    if any(not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]*", name) for name in names):
        raise ValueError("Instance and submission names must be simple filenames; model may contain a repository path")
    return data


def prepare(request_path, output_dir, *, repository=None):
    import qoblib

    request = read_request(request_path)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise FileExistsError(f"Benchmark bundle already exists: {root}")
    source_dir = root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    ref = repository_ref(repository) if repository else request.get("ref", qoblib.get_ref())
    records = []
    seen = set()
    entries = []
    for entry in request["instances"]:
        if entry["problem"] != "birkhoff":
            entries.append(entry)
            continue
        name = str(entry["instance"])
        problem = get_problem("birkhoff", ref=ref, source_dir=source_dir, repository=repository)
        collection = load_instance(problem, name)
        selected = entry.get("subinstance")
        matches = [(k, v) for k, v in collection.instances.items() if selected is None or selected in {k, v["id"]}]
        if not matches:
            raise ValueError(f"No Birkhoff sub-instance {selected!r} in {name}")
        for key, data in matches:
            entries.append({**entry, "instance": data["id"], "source_instance": name, "subinstance": key})
    for entry in entries:
        slug, name = entry["problem"], str(entry["instance"])
        if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]*", name):
            raise ValueError("Invalid instance identifier in source data")
        if slug not in DIRECTORIES:
            raise ValueError(f"Unknown QOBLIB problem: {slug}")
        problem = get_problem(slug, ref=ref, source_dir=source_dir, repository=repository)
        source_name = entry.get("source_instance", name)
        instance = load_instance(problem, source_name)
        if slug == "portfolio":
            parameters = entry.get("parameters", {})
            if "budget" not in parameters or "lambda" not in parameters:
                raise ValueError("Portfolio requires parameters.budget and parameters.lambda matching the LP variant")
        if slug == "birkhoff":
            from qoblib.problems.birkhoff import BirkhoffInstance

            key = entry["subinstance"]
            instance = BirkhoffInstance({key: instance.instances[key]}, name=name, path=instance.path)
        options = {
            "penalty": entry.get("penalty", request.get("penalty")),
            "max_variables": request.get("max_variables", 4096),
        }
        if slug == "labs":
            record = labs_record(instance, **options)
            formulation = "native_labs_products"
            name = f"labs{instance:03d}"
        else:
            if slug in NATIVE:
                if entry.get("model"):
                    raise ValueError(f"{slug} uses its native reversible formulation; omit model")
                model = native_model(slug, instance, max_variables=options["max_variables"])
                formulation = f"native_{slug}"
            else:
                formulation = entry.get("formulation", DEFAULT_FORMULATION[slug])
                path = load_model(problem, entry.get("model", name), formulation)
                model = read_lp(path)
                if slug == "portfolio":
                    name = path.stem
                for variable, bounds in entry.get("bounds", {}).items():
                    if variable not in model["variables"]:
                        raise ValueError(f"Unknown bound override: {variable}")
                    model["variables"][variable] = bounds
                if any(abs(v) >= 1e20 for bounds in model["variables"].values() for v in bounds):
                    raise ValueError("Unbounded model variable: provide explicit finite bounds in the request")
            record = encode_model(model, **options)
        if (slug, name) in seen:
            raise ValueError(f"Duplicate instance: {slug}/{name}")
        seen.add((slug, name))
        fingerprint = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:16]
        safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", name)
        identity = f"{slug}_{safe_name}_{fingerprint}"
        reference = None
        reference_name = entry.get("reference_solution", name if slug == "portfolio" else source_name)
        try:
            if slug == "labs":
                from qoblib.problems.labs import LABSSolution

                path = problem.fetch(entry.get("reference_solution", f"labs{instance:03d}"), kind="solution")
                text = "".join(line.strip() for line in path.read_text().splitlines() if not line.startswith("#"))
                if len(text) != instance or set(text) - {"0", "1"}:
                    raise ValueError("Unrecognized LABS reference sequence")
                reference = LABSSolution([2 * int(v) - 1 for v in text])
            else:
                reference = problem.load_solution(reference_name)
            if slug == "birkhoff":
                from qoblib.problems.birkhoff import BirkhoffSolution

                reference = BirkhoffSolution({k: v for k, v in reference.solutions.items() if v["id"] == name})
                if not reference.solutions:
                    raise ValueError(f"No Birkhoff reference solution for {name}")
            if slug == "sports":
                reference = None
        except (KeyError, ValueError, NotImplementedError, qoblib.QoblibError) as exc:
            reference_error = str(exc)
        else:
            reference_error = None
        reference_objective = problem.compute_objective(instance, reference) if reference is not None else None
        record.update(
            {
                "problem": slug,
                "instance": name,
                "source_instance": source_name,
                "subinstance": entry.get("subinstance"),
                "instance_id": identity,
                "ref": ref,
                "formulation": formulation,
                "parameters": entry.get("parameters", {}),
                "variable_aliases": entry.get("variable_aliases", {}),
                "reference_objective": reference_objective,
                "reference_error": reference_error,
            }
        )
        if record["dim"] ** 2 * 8 > request.get("max_matrix_bytes", 256 * 1024**2):
            raise ValueError(f"{name}: dense QSplit matrix exceeds max_matrix_bytes")
        for run in range(request.get("runs", 5)):
            records.append({**record, "id": f"{identity}_run{run:03d}", "run": run})
    dataset = root / "dataset.jsonl"
    dataset.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records), encoding="utf-8")
    write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "ref": ref,
            "request": request,
            "source_hashes": source_hashes(source_dir),
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "count": len(records),
            "versions": {p: version(p) for p in ["qsplit", "qoblib", "numpy", "pyscipopt"]},
        },
    )
    return dataset


def solution_object(record, instance, values):
    slug = record["problem"]
    if slug == "marketsplit":
        from qoblib.problems.marketsplit import MarketSplitSolution

        return MarketSplitSolution([values[f"x#{i + 1}"] for i in range(instance.n_cols)]), ".sol"
    if slug == "independentset":
        from qoblib.problems.independentset import StableSetSolution

        return StableSetSolution({i + 1 for i in range(instance.n) if values[f"x#{i + 1}"]}), ".sol"
    if slug == "labs":
        from qoblib.problems.labs import LABSSolution

        return LABSSolution([2 * values[f"x#{i}"] - 1 for i in range(instance)]), ".sol"
    if slug == "birkhoff":
        from qoblib.problems.birkhoff import BirkhoffSolution

        data = {}
        for key, permutations in record["bridge"]["model"]["permutations"].items():
            active = [(p, values[f"w${key}#{i}"]) for i, p in enumerate(permutations) if values[f"w${key}#{i}"]]
            data[key] = {
                "id": instance.instances[key]["id"],
                "permutations": [v for p, _ in active for v in p],
                "weights": [w for _, w in active],
            }
        return BirkhoffSolution(data), ".json"
    if slug == "steiner":
        from qoblib.problems.steiner import SteinerSolution

        edges = set()
        for name, value in values.items():
            if value and (match := re.fullmatch(r"y#(\d+)#(\d+)#(\d+)", name)):
                u, v, net = map(int, match.groups())
                edges.add((min(u, v), max(u, v), net))
        return SteinerSolution(sorted(edges)), ".sol"
    if slug == "network":
        from qoblib.problems.network import NetworkSolution

        edges, flows = {}, {}
        for name, value in values.items():
            if match := re.fullmatch(r"x#(\d+)#(\d+)", name):
                edges[tuple(map(int, match.groups()))] = value
            if match := re.fullmatch(r"f#(\d+)#(\d+)#(\d+)", name):
                flows[tuple(map(int, match.groups()))] = value
        return NetworkSolution(objective=values["z"], edges=edges, flows=flows), ".sol"
    if slug == "routing":
        from qoblib.problems.routing import CVRPSolution

        arcs = {
            (int(m[1]), int(m[2]))
            for name, v in values.items()
            if v and (m := re.fullmatch(r"x#(\d+)#(\d+)", name)) and m[1] != m[2]
        }
        routes, visited = [], set()
        for start in sorted(v for u, v in arcs if u == instance.depot):
            route, node = [], start
            while node != instance.depot:
                if node in visited:
                    raise ValueError("Routing assignment has a cycle or repeated customer")
                visited.add(node)
                route.append(node)
                successors = [v for u, v in arcs if u == node]
                if len(successors) != 1:
                    raise ValueError("Routing assignment has missing or multiple successors")
                node = successors[0]
            routes.append(route)
        if visited != set(instance.coords) - {instance.depot}:
            raise ValueError("Routing assignment does not connect every customer to the depot")
        return CVRPSolution(routes), ".sol"
    if slug == "topology":
        from qoblib.problems.topology import TopologySolution

        edges = set()
        for name, value in values.items():
            if not value:
                continue
            if match := re.fullmatch(r"dist#(\d+)#(\d+)#0", name):
                edges.add(tuple(int(v) + 1 for v in match.groups()))
            elif match := re.fullmatch(r"z#(\d+)#(\d+)", name):
                edges.add(tuple(int(v) + 1 for v in match.groups()))
        solution = TopologySolution(instance.n, frozenset(edges), degree=instance.degree)
        from qoblib import get_problem as qoblib_problem

        diameter = qoblib_problem("topology").compute_objective(instance, solution)
        solution.diameter = int(diameter) if diameter is not None else instance.n
        return solution, ".gph"
    if slug == "sports":
        root = ET.Element("Solution")
        meta = ET.SubElement(root, "MetaData")
        ET.SubElement(meta, "InstanceName").text = record["instance"] + ".xml"
        ET.SubElement(meta, "SolutionMethod").text = "QSplit"
        games = ET.SubElement(root, "Games")
        for name, value in values.items():
            if value and (match := re.fullmatch(r"x#(\d+)#(\d+)#(\d+)", name)):
                ET.SubElement(games, "ScheduledMatch", dict(zip(["home", "away", "slot"], match.groups())))
        return ET.tostring(root, encoding="unicode"), ".xml"
    if slug == "portfolio":
        from qoblib.problems.portfolio import PortfolioPosition, PortfolioSolution

        parameters = record["parameters"]
        if "budget" not in parameters or "lambda" not in parameters:
            raise ValueError("Portfolio requires parameters.budget and parameters.lambda matching the LP variant")
        full = [re.fullmatch(r"x\$([^#]+)#(\d+)#1#(\d+)", v) for v in values]
        full = [m for m in full if m]
        if not full:
            raise ValueError("Unrecognized portfolio variable names")
        periods = max(int(m[3]) for m in full) + 1
        units = max(int(m[2]) for m in full)
        positions = defaultdict(lambda: [0, 0])
        for name, value in values.items():
            if not name.startswith("x$"):
                continue
            alias = record["variable_aliases"].get(name, name)
            match = re.fullmatch(r"x\$([^#]+)#(\d+)#(1|_1|-1)#(\d+)", alias)
            if match:
                symbol, _, sign, period = match.groups()
                slot = 0 if sign == "1" else 1
            elif match := re.fullmatch(r"x\$([^#]+)#.*@([0-9a-f]+)", alias):
                symbol, ordinal = match.groups()
                ordinal = int(ordinal, 16)
                period, slot = ordinal % periods, (ordinal // periods) % 2
            else:
                raise ValueError(f"Unrecognized portfolio variable {name}; provide variable_aliases")
            if units < 1:
                raise ValueError("Invalid portfolio unit layout")
            positions[(int(period), symbol)][slot] += value
        rows = [PortfolioPosition(t, s, *v) for (t, s), v in sorted(positions.items())]
        objective, _ = model_evaluation(record["bridge"]["model"], values)
        return PortfolioSolution(
            record.get("source_instance", record["instance"]),
            parameters["budget"],
            parameters["lambda"],
            objective,
            rows,
        ), ".sol"
    raise ValueError(f"Unsupported solution format for {slug}")


def serialize(solution, slug, instance):
    if isinstance(solution, str):
        return solution + "\n"
    if slug == "independentset":
        return "".join("1" if i in solution.selected else "0" for i in range(1, instance.n + 1)) + "\n"
    return solution.to_checker_string().rstrip() + "\n"


def finalize(bundle_dir, solutions_dir, output_dir, *, timings=None, pipeline=None):
    import qoblib

    bundle, output = Path(bundle_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported benchmark schema")
    if source_hashes(bundle / "source") != manifest["source_hashes"]:
        raise ValueError("Benchmark source files changed since preparation")
    if hashlib.sha256((bundle / "dataset.jsonl").read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Benchmark dataset changed since preparation")
    request = manifest["request"]
    rows = []
    records = [json.loads(line) for line in (bundle / "dataset.jsonl").read_text().splitlines() if line.strip()]
    problems = {}
    for record in records:
        slug, name = record["problem"], record["instance"]
        if slug not in problems:
            problems[slug] = get_problem(
                slug, ref=record["ref"], source_dir=bundle / "source", repository=bundle / "source"
            )
        problem = problems[slug]
        instance = load_instance(problem, record["source_instance"])
        if slug == "birkhoff":
            from qoblib.problems.birkhoff import BirkhoffInstance

            key = record["subinstance"]
            instance = BirkhoffInstance({key: instance.instances[key]}, name=name, path=instance.path)
        path = Path(solutions_dir) / f"solutions_{record['id']}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed run: {path}")
        with path.open(newline="") as stream:
            candidates = [r for r in csv.DictReader(stream) if r["node_id"] == "root" and r["backend"] == "aggregate"]
        if not candidates:
            raise ValueError(f"No complete root aggregate solution in {path}")
        evaluated = []
        for candidate in candidates:
            bits = candidate["bitstring"]
            values = decode(record, bits)
            matrix_energy = sum(v * int(bits[i]) * int(bits[j]) for i, j, v in record["qubo_mat"])
            energy = matrix_energy + record["offset"]
            if record["bridge"]["kind"] == "labs":
                objective = sum(
                    sum((2 * values[f"x#{i}"] - 1) * (2 * values[f"x#{i + k}"] - 1) for i in range(instance - k)) ** 2
                    for k in range(1, instance)
                )
                violations = []
            else:
                objective, violations = model_evaluation(record["bridge"]["model"], values)
            target_dir = output / DIRECTORIES[slug] / "submissions" / request.get("submission_name", "QSplit") / name
            target_dir.mkdir(parents=True, exist_ok=True)
            row = {
                "id": record["id"],
                "problem": slug,
                "instance": name,
                "run": record["run"],
                "matrix_energy": matrix_energy,
                "qubo_energy": energy,
                "offset": record["offset"],
                "model_objective": objective,
                "model_feasible": not violations,
                "model_violations": violations,
                "objective": None,
                "feasible": None,
                "checker_status": "NOT_CHECKED",
                "checker": None,
                "decode_error": None,
                "solution_file": None,
                "submission_ready": False,
                "runtime_seconds": (timings or {}).get(record["id"]),
            }
            try:
                solution, extension = solution_object(record, instance, values)
                solution_path = target_dir / f"{name}_run{record['run']:03d}_solution{extension}"
                solution_path.write_text(serialize(solution, slug, instance), encoding="utf-8")
                row["solution_file"] = solution_path.relative_to(output).as_posix()
                if slug != "sports":
                    row["objective"] = problem.compute_objective(instance, solution)
                if request.get("check", "auto") != "off":
                    try:
                        result = problem.check_solution(instance, solution)
                        row.update(feasible=result.feasible, checker_status=result.status, checker=asdict(result))
                        row["submission_ready"] = result.feasible and not violations
                    except qoblib.QoblibError as exc:
                        row.update(checker_status="UNAVAILABLE", checker={"error": str(exc)})
                        if request.get("check") == "required":
                            raise
            except ValueError as exc:
                row["decode_error"] = str(exc)
                row["checker_status"] = "DECODE_FAILED"
            row["reference_objective"] = record["reference_objective"]
            reference = record["reference_objective"]
            maximize = slug == "independentset"
            row["improvement"] = (
                None
                if reference is None or row["objective"] is None or row["feasible"] is not True
                else (row["objective"] - reference if maximize else reference - row["objective"])
            )
            evaluated.append(row)

        def rank(row):
            score = row["objective"] if row["objective"] is not None else row["model_objective"]
            return (
                row["feasible"] is not True,
                not row["model_feasible"],
                row["decode_error"] is not None,
                -score if slug == "independentset" else score,
                row["qubo_energy"],
            )

        best = min(evaluated, key=rank)
        rows.append(best)
        write_json(target_dir / f"{name}_run{record['run']:03d}_result.json", best)
        write_json(
            target_dir / f"{name}_run{record['run']:03d}_variables.json",
            decode(record, candidates[evaluated.index(best)]["bitstring"]),
        )
    groups = defaultdict(list)
    for record, row in zip(records, rows):
        groups[(row["problem"], row["instance"])].append((record, row))
    summaries = []
    for (slug, name), runs in groups.items():
        record = runs[0][0]
        feasible = [r for _, r in runs if r["feasible"] is True and r["model_feasible"] and r["objective"] is not None]
        best = (max if slug == "independentset" else min)(feasible, key=lambda r: r["objective"]) if feasible else None
        summary = dict.fromkeys(SUBMISSION_COLUMNS, "N/A")
        submission = request.get("submission", {})
        summary.update(
            {
                "Problem": name,
                "Submitter": submission.get("submitter", "N/A"),
                "Affiliation": submission.get("affiliation", "N/A"),
                "Date": submission.get("date", datetime.now(ZoneInfo("Europe/Rome")).date().isoformat()),
                "Reference": submission.get("reference", "N/A"),
                "Best Objective Value": best["objective"] if best else "N/A",
                "Modeling Approach": record["formulation"] + " -> QUBO",
                "# Decision Variables": record["dim"],
                "# Binary Variables": record["dim"],
                "# Integer Variables": 0,
                "# Continuous Variables": 0,
                "# Non-Zero Coefficients": len(record["qubo_mat"]),
                "Coefficients Type": "Real",
                "Coefficients Range": str(
                    [
                        min((t[2] for t in record["qubo_mat"]), default=0),
                        max((t[2] for t in record["qubo_mat"]), default=0),
                    ]
                ),
                "Workflow": "QOBLIB bridge -> QSplit split/solve/aggregate/refine -> original solution -> checker",
                "Algorithm Type": "Stochastic",
                "Paradigm": submission.get("paradigm", "Classical"),
                "# Runs": len(runs),
                "# Feasible Runs": len(feasible),
                "# Successful Runs": sum(r["objective"] == best["objective"] for r in feasible) if best else 0,
                "Success Threshold": 0,
                "Hardware Specifications": submission.get("hardware", platform.platform()),
                "Remarks": f"DRAFT; QOBLIB ref {record['ref']}; unverified runs are not counted as feasible",
            }
        )
        runtime = [r["runtime_seconds"] for _, r in runs]
        for field in [
            "Total Runtime",
            "Time to Solution",
            "CPU Runtime",
            "GPU Runtime",
            "QPU Runtime",
            "Other HW Runtime",
        ]:
            summary[field] = submission.get("runtimes", {}).get(field, "N/A")
        target = output / DIRECTORIES[slug] / "submissions" / request.get("submission_name", "QSplit") / name
        with (target / f"{name}_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUBMISSION_COLUMNS)
            writer.writeheader()
            writer.writerow(summary)
        summaries.append(
            {
                "problem": slug,
                "instance": name,
                "runs": len(runs),
                "feasible_runs": len(feasible),
                "best_objective": best["objective"] if best else None,
                "reference_objective": record["reference_objective"],
                "mean_pipeline_seconds": sum(runtime) / len(runtime) if all(t is not None for t in runtime) else None,
                "submission_ready": bool(best)
                and len(runs) >= 5
                and summary["Total Runtime"] != "N/A"
                and all(submission.get(k) for k in ["submitter", "affiliation", "reference", "hardware"]),
            }
        )
    write_json(
        output / "results.json", {"schema_version": 1, "ref": manifest["ref"], "runs": rows, "summary": summaries}
    )
    write_json(output / "provenance.json", {**manifest, "pipeline": pipeline})
    (output / "README.md").write_text(
        "# QSplit QOBLIB benchmark\n\nSummary CSVs are submission drafts. Only official checker verdicts count as "
        "feasible runs. Review author, affiliation, hardware, reference, full runtime accounting and the upstream "
        "submission validator before submitting. QUBO energies include penalties and offsets; compare original "
        "objectives in results.json. Source hashes and versions are in provenance.json.\n",
        encoding="utf-8",
    )
    return output / "results.json"
