import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _looks_like_project_root(path: Path) -> bool:
    return (path / "qsplit").is_dir() and (path / "streamflow").is_dir()


def _candidate_launch_dirs() -> list[Path]:
    candidates: list[Path] = []
    for env_name in ("QSPLIT_LAUNCH_DIR", "QSPLIT_PROJECT_ROOT", "PWD", "OLDPWD", "INIT_CWD"):
        raw = os.getenv(env_name, "").strip()
        if not raw:
            continue
        candidates.append(Path(raw).expanduser())
    return candidates


def _is_ephemeral_solutions_dir(path: Path) -> bool:
    normalized = str(path.resolve())
    return "/tmp/streamflow/" in normalized or "/private/tmp/streamflow/" in normalized


def _resolve_solutions_dir(raw: str) -> Path:
    base = Path(raw).expanduser()
    if base.is_absolute():
        return base

    for candidate in _candidate_launch_dirs():
        resolved = candidate.resolve()
        if _looks_like_project_root(resolved):
            return resolved / base

    repo_root = _repo_root().resolve()
    if _looks_like_project_root(repo_root):
        return repo_root / base

    for candidate in _candidate_launch_dirs():
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved / base

    return Path.cwd().resolve() / base


def _is_valid_solution_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if path.stat().st_size <= 0:
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            first_row = next(reader, None)
    except Exception:
        return False
    if header != ["node_id", "backend", "bitstring", "energy"]:
        return False
    if not first_row or len(first_row) != 4:
        return False
    try:
        float(first_row[3])
    except Exception:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect per-instance QSplit solutions for a dataset run.")
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--solutions-dir", default="solutions")
    parser.add_argument("--solution-csv", action="append", nargs="+", default=[])
    parser.add_argument("--output-dir", default="solutions_dataset")
    parser.add_argument("--output-manifest", default="dataset_results_manifest.json")
    args = parser.parse_args()

    manifest_path = Path(args.dataset_manifest).resolve()
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = dataset_manifest.get("items", [])
    solutions_dir = _resolve_solutions_dir(args.solutions_dir)
    print(f"QSPLIT COLLECT solutions_dir={solutions_dir}", flush=True)
    if _is_ephemeral_solutions_dir(solutions_dir):
        print(
            "QSPLIT WARNING solutions_dir points to a temporary StreamFlow directory. "
            "Use an absolute path in cwl/config.yml (solutions_store_dir).",
            flush=True,
        )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.csv"):
        stale.unlink()

    results: list[dict] = []
    for item in items:
        output_name = item.get("solution_csv") or f"solutions_{item['safe_id']}.csv"
        source = solutions_dir / output_name
        if _is_valid_solution_file(source):
            target = output_dir / output_name
            shutil.copy2(source, target)
            status = "ok"
        else:
            output_name = None
            status = "missing"
        results.append(
            {
                "index": item["index"],
                "line": item["line"],
                "id": item["id"],
                "safe_id": item["safe_id"],
                "matrix_csv": item["matrix_csv"],
                "solution_csv": output_name,
                "status": status,
            }
        )

    output_manifest = Path(args.output_manifest).resolve()
    output_manifest.write_text(
        json.dumps(
            {
                "dataset": dataset_manifest.get("dataset"),
                "count": len(items),
                "resolved_solutions": sum(1 for row in results if row["status"] == "ok"),
                "solutions_dir": str(solutions_dir),
                "items": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
