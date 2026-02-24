import argparse
import csv
import json
import os
import re
from pathlib import Path


def _record_id(record: dict, line_no: int) -> str:
    raw = record.get("id")
    if raw is None:
        raw = record.get("originale_index", line_no)
    value = str(raw).strip()
    return value if value else str(line_no)


def _safe_id(value: str, fallback: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return safe if safe else f"row_{fallback}"


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


def _expected_solution_name(safe_id: str) -> str:
    return f"solutions_{safe_id}.csv"


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


def _matrix_from_record(record: dict) -> list[list[float]] | None:
    try:
        dim = int(record.get("dim", 0))
    except (TypeError, ValueError):
        return None
    if dim <= 0:
        return None

    matrix = [[0.0 for _ in range(dim)] for _ in range(dim)]
    for term in record.get("qubo_mat", []):
        if not isinstance(term, (list, tuple)) or len(term) != 3:
            continue
        try:
            i = int(term[0])
            j = int(term[1])
            value = float(term[2])
        except (TypeError, ValueError):
            continue
        if i < 0 or j < 0 or i >= dim or j >= dim:
            continue
        if i <= j:
            matrix[i][j] = value
        else:
            matrix[j][i] = value
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare per-instance matrix CSV files from a JSONL dataset.")
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--output-dir", default="dataset_matrices")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--solutions-dir", default="solutions")
    args = parser.parse_args()

    if args.max_instances < 0:
        raise ValueError("--max-instances must be >= 0.")

    dataset_path = Path(args.dataset_jsonl).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.csv"):
        stale.unlink()
    solutions_dir = _resolve_solutions_dir(args.solutions_dir)
    solutions_dir.mkdir(parents=True, exist_ok=True)
    print(f"QSPLIT DATASET PREPARE solutions_dir={solutions_dir}", flush=True)
    if _is_ephemeral_solutions_dir(solutions_dir):
        print(
            "QSPLIT WARNING solutions_dir points to a temporary StreamFlow directory. "
            "Use an absolute path in cwl/config.yml (solutions_store_dir).",
            flush=True,
        )

    manifest_items: list[dict] = []
    scheduled_count = 0
    with dataset_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if args.max_instances > 0 and len(manifest_items) >= args.max_instances:
                break
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue

            matrix = _matrix_from_record(record)
            if matrix is None:
                continue

            rec_id = _record_id(record, line_no)
            safe_id = _safe_id(rec_id, line_no)
            solution_name = _expected_solution_name(safe_id)
            solution_path = solutions_dir / solution_name
            already_solved = _is_valid_solution_file(solution_path)
            matrix_name = None

            if not already_solved:
                matrix_name = f"{scheduled_count:06d}_{safe_id}.csv"
                matrix_path = output_dir / matrix_name
                matrix_path.write_text(
                    "\n".join(",".join(f"{value:g}" for value in row) for row in matrix) + "\n",
                    encoding="utf-8",
                )
                scheduled_count += 1

            manifest_items.append(
                {
                    "index": len(manifest_items),
                    "line": line_no,
                    "id": rec_id,
                    "safe_id": safe_id,
                    "solution_csv": solution_name,
                    "already_solved": already_solved,
                    "matrix_csv": matrix_name,
                }
            )

    if not manifest_items:
        raise SystemExit("No valid records processed from dataset.")

    manifest_path = Path(args.manifest).resolve()
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "count": len(manifest_items),
                "scheduled_count": scheduled_count,
                "solutions_dir": str(solutions_dir),
                "items": manifest_items,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"QSPLIT DATASET PREPARE processed={len(manifest_items)} scheduled={scheduled_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
