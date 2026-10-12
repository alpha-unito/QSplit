#!/usr/bin/env python3
"""
.venv/bin/python qoblib_stats.py --repository ../QOBLIB
.venv/bin/python qoblib_stats.py --repository ../QOBLIB --only-valid
"""

import argparse
import csv
import html
import json
import math
import os
import statistics
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager, redirect_stderr
from pathlib import Path

from qsplit.benchmark.bridge import DEFAULT_FORMULATION, read_request
from qsplit.benchmark.conversion import model_qubo_dimension as qubo_size
from qsplit.benchmark.conversion import read_lp
from qsplit.benchmark.sources import (
    DIRECTORIES,
    catalog_entries,
    decompress,
    get_problem,
    load_instance,
    repository_ref,
)


@contextmanager
def quiet_stderr():
    with open(os.devnull, "w") as quiet, redirect_stderr(quiet):
        previous = os.dup(2)
        try:
            os.dup2(quiet.fileno(), 2)
            yield
        finally:
            os.dup2(previous, 2)
            os.close(previous)


def census(repository, scratch, *, limits=None):
    rows = []
    skipped = 0
    variable_limit = (
        None if limits is None else min(limits["max_variables"], math.isqrt(limits["max_matrix_bytes"] // 8))
    )

    def collect(row):
        nonlocal skipped
        if limits is not None and (
            row["error"]
            or row["variables"] is None
            or row["variables"] < 1
            or row["qubo_variables"] is None
            or not 1 <= row["qubo_variables"] <= variable_limit
        ):
            skipped += 1
        else:
            rows.append(row)

    entries = catalog_entries(repository, DEFAULT_FORMULATION)
    ref = repository_ref(repository)
    for position, entry in enumerate(entries, 1):
        instance = model = None
        slug, name = entry["problem"], entry["instance"]
        if limits is None:
            print(f"[{position}/{len(entries)}] {slug}/{name}", file=sys.stderr, flush=True)
        row = {
            "problem": slug,
            "instance": name,
            "model": entry.get("model", ""),
            "variables": None,
            "qubo_variables": None,
            "error": "",
        }
        path = None
        try:
            if slug in {"marketsplit", "labs", "independentset", "birkhoff"}:
                problem = get_problem(slug, ref=ref, source_dir=scratch, repository=repository)
                instance = load_instance(problem, entry.get("source_instance", name))
                if slug == "birkhoff":
                    for key, data in instance.instances.items():
                        try:
                            if limits is not None and (int(data["n"]) < 1 or int(data["scale"]) < 0):
                                raise ValueError("Invalid Birkhoff domain")
                            permutations = math.factorial(int(data["n"]))
                            bits = int(data["scale"]).bit_length()
                            dimension = permutations * (1 + 2 * bits)
                            if limits is not None and dimension <= variable_limit:
                                matrix = data["scaled_doubly_stochastic_matrix"]
                                if len(matrix) < int(data["n"]) ** 2 or any(not math.isfinite(v) for v in matrix):
                                    raise ValueError("Invalid Birkhoff matrix")
                            collect(
                                {
                                    **row,
                                    "instance": data["id"],
                                    "variables": 2 * permutations,
                                    "qubo_variables": dimension,
                                }
                            )
                        except Exception as exc:
                            collect({**row, "instance": key, "error": str(exc)})
                    continue
                if slug == "labs":
                    row.update(variables=instance, qubo_variables=instance * (instance + 1) // 2)
                elif slug == "marketsplit":
                    row.update(variables=instance.n_cols, qubo_variables=instance.n_cols)
                else:
                    row.update(variables=instance.n, qubo_variables=max(1, instance.n + len(instance.edges)))
            else:
                if "model" not in entry:
                    raise ValueError(f"Missing {DEFAULT_FORMULATION[slug]} model")
                path = decompress(repository / entry["model"], scratch / "lp")
                model = read_lp(path, max_variables=variable_limit)
                row["variables"] = len(model["variables"])
                if any(abs(v) >= 1e20 for bounds in model["variables"].values() for v in bounds):
                    raise ValueError("Unbounded model variable")
                row["qubo_variables"] = qubo_size(model, max_variables=variable_limit)
                if limits is not None:
                    problem = get_problem(slug, ref=ref, source_dir=scratch, repository=repository)
                    load_instance(problem, name)
        except Exception as exc:
            row["error"] = str(exc)
            # Unsupported continuous/nonlinear LPs still have a measurable size.
            if limits is None and path is not None and row["variables"] is None:
                try:
                    from pyscipopt import Model

                    scip = Model()
                    scip.hideOutput()
                    scip.readProblem(str(path))
                    row["variables"] = sum(v.name != "quadobjvar" for v in scip.getVars())
                except Exception as count_error:
                    row["error"] += f"; LP variable count failed: {count_error}"
        finally:
            if path is not None and path != repository / entry.get("model", ""):
                path.unlink(missing_ok=True)
        collect(row)
    return rows, skipped


def describe(values):
    if not values:
        return {"count": 0}
    frequencies = Counter(values)
    frequency = max(frequencies.values())
    modes = sorted(v for v, count in frequencies.items() if count == frequency)
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "modes": modes if frequency > 1 else [],
        "mode_frequency": frequency,
    }


def histogram(values):
    bins = Counter(-1 if v == 0 else len(str(v)) - 1 for v in values)
    if not bins:
        return "<p>No dimensions available.</p>"
    bars = []
    for exponent in range(min(bins), max(bins) + 1):
        count = bins[exponent]
        label = "0" if exponent == -1 else f"10^{exponent} ≤ n &lt; 10^{exponent + 1}"
        bars.append(
            f'<div class="bin"><span>{label}</span><div class="track"><i style="width:'
            f'{100 * count / max(bins.values()):.2f}%"></i></div><b>{count}</b></div>'
        )
    return "".join(bars)


def write_report(rows, output, repository, *, only_valid=False):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "instances.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["problem", "instance", "model", "variables", "qubo_variables", "error"]
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "repository": str(repository),
        "ref": repository_ref(repository),
        "catalog_count": len(rows),
        "only_valid": only_valid,
        "notes": "Theoretical sizes of QSplit formulations; execution feasibility is not checked. "
        "Birkhoff subinstances expanded; portfolio variants counted separately. Statistics use known sizes only.",
        "statistics": {},
        "by_problem": {},
    }
    if only_valid:
        summary["notes"] = (
            "Only instances compatible with the benchmark formulations and QUBO limits. "
            "Birkhoff subinstances expanded; portfolio variants counted separately. "
            "No solving or feasibility checks performed."
        )
    for key in ("variables", "qubo_variables"):
        summary["statistics"][key] = describe([r[key] for r in rows if r[key] is not None])
    for slug in DIRECTORIES:
        selected = [r for r in rows if r["problem"] == slug]
        if only_valid and not selected:
            continue
        summary["by_problem"][slug] = {
            "count": len(selected),
            **{
                key: describe([r[key] for r in selected if r[key] is not None])
                for key in ("variables", "qubo_variables")
            },
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    sections = []
    for key, title in (("variables", "Formulation variables"), ("qubo_variables", "QUBO variables")):
        stats = summary["statistics"][key]
        print(f"\n{title}: {json.dumps(stats)}")
        body = "".join(
            f"<tr><th>{label}</th><td>{html.escape(str(stats.get(field, '—')))}</td></tr>"
            for field, label in (
                ("count", "Measured instances"),
                ("mean", "Mean"),
                ("min", "Minimum"),
                ("max", "Maximum"),
                ("median", "Median"),
                ("modes", "Modes"),
                ("mode_frequency", "Mode frequency"),
            )
        )
        sections.append(
            f"<h2>{title}</h2><p>Unknown size: {len(rows) - stats['count']} instances. "
            "An empty modes list means no value repeats.</p><table>" + body + "</table><h3>Histogram</h3>"
            "<p>Bins by order of magnitude; bar length represents the number of instances.</p>"
            + histogram([r[key] for r in rows if r[key] is not None])
        )
    counts = "".join(
        f"<tr><td>{slug}</td><td>{data['count']}</td><td>{data['variables']['count']}</td>"
        f"<td>{data['qubo_variables']['count']}</td></tr>"
        for slug, data in summary["by_problem"].items()
    )
    coverage = (
        "The report includes only compatible instances within the benchmark limits."
        if only_valid
        else "Unsupported LPs and missing models remain in the total."
    )
    report = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>QOBLIB dataset sizes</title>
<style>body{{font:16px system-ui;margin:40px auto;max-width:1100px;padding:0 24px;color:#182337}}
table{{border-collapse:collapse;display:block;overflow:auto}}
td,th{{padding:8px;text-align:left;border-bottom:1px solid #ddd}}
.bin{{display:grid;grid-template-columns:230px 1fr 60px;gap:12px;margin:4px 0;font-size:13px}}
.track{{background:#f1f4f8}}i{{display:block;background:#3676b9;height:19px}}h2{{margin-top:48px}}</style>
<h1>QOBLIB: {len(rows)} instances</h1><p>{html.escape(summary["notes"])}</p>
<p>Formulation variables come from selected LP models or the bridge's native formulations.
QUBO dimensions include bits, LABS product variables, and slack.
The native Birkhoff formulation uses 2·n! decision variables; its large sizes reflect this formulation.
{coverage} See <a href="instances.csv">instances.csv</a>
for sizes and errors, and <a href="summary.json">summary.json</a> for statistics, including per-class results.</p>
<table><tr><th>Class</th><th>Instances</th><th>Known formulation sizes</th><th>Known QUBO sizes</th></tr>
{counts}</table>{"".join(sections)}</html>"""
    (output / "report.html").write_text(report, encoding="utf-8")
    print(f"\nTotal instances: {len(rows)}\nReport: {output / 'report.html'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parent.parent / "QOBLIB")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--only-valid", action="store_true", help="Exclude invalid instances and those exceeding benchmark limits"
    )
    parser.add_argument(
        "--self-check", action="store_true", help="Check variable counts on small models without reading a dataset"
    )
    args = parser.parse_args()
    if args.self_check:
        from qsplit.benchmark.conversion import encode_model

        for coefficients, lower, upper in (
            ({"x": 2, "y": -1}, -3, 5),
            ({"x": 0.5, "y": -0.25}, -0.75, 1.25),
            ({"x": 1}, 1, 1),
        ):
            model = {
                "variables": {"x": [-2, 5], "y": [0, 1]},
                "constraints": [{"coefficients": coefficients, "lower": lower, "upper": upper}],
            }
            assert qubo_size(model) == encode_model(model)["dim"]
        assert describe([1, 1, 2, 2, 3])["modes"] == [1, 2]
        assert describe([]) == {"count": 0}
        print("Variable counts verified against encode_model on small models.")
        return
    limits = (
        read_request(Path(__file__).resolve().parent / "streamflow/cwl/benchmark.all.request.template.yaml")
        if args.only_valid
        else None
    )
    output = args.output_dir or Path("reports/qoblib-stats-valid" if args.only_valid else "reports/qoblib-stats")
    with tempfile.TemporaryDirectory(prefix="qoblib-stats-") as temporary:
        if args.only_valid:
            with quiet_stderr():
                rows, skipped = census(args.repository.resolve(), Path(temporary), limits=limits)
        else:
            rows, skipped = census(args.repository.resolve(), Path(temporary))
    write_report(rows, output.resolve(), args.repository.resolve(), only_valid=args.only_valid)
    if args.only_valid:
        print(f"Skipped instances: {skipped}")


if __name__ == "__main__":
    main()
