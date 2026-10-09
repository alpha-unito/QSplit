import argparse
import re
import shutil
from pathlib import Path

from qsplit import configuration


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _looks_like_project_root(path: Path) -> bool:
    return (path / "qsplit").is_dir() and (path / "streamflow").is_dir()


def _candidate_launch_dirs() -> list[Path]:
    candidates: list[Path] = []
    for key in ("QSPLIT_LAUNCH_DIR", "QSPLIT_PROJECT_ROOT"):
        raw = configuration.get(key, "").strip()
        if not raw:
            continue
        candidates.append(Path(raw).expanduser())
    return candidates


def _is_ephemeral_solutions_dir(path: Path) -> bool:
    normalized = path.resolve().as_posix()
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


def _safe_id_from_matrix(path: Path) -> str:
    stem = path.stem
    while True:
        match = re.match(r"^\d{6}_(.+)$", stem)
        if not match:
            break
        stem = match.group(1)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return safe or "instance_unknown"


@configuration.cli
def main() -> None:
    parser = argparse.ArgumentParser(description="Persist a completed instance solution to the shared solutions dir.")
    parser.add_argument("--input-solution", required=True)
    parser.add_argument("--input-matrix", required=True)
    parser.add_argument("--solutions-dir", default="solutions")
    parser.add_argument("--output-solution", default="persisted_solution.csv")
    parser.add_argument("--config", action="append", help="YAML configuration file; repeat to layer files")
    args = parser.parse_args()

    input_solution = Path(args.input_solution).resolve()
    input_matrix = Path(args.input_matrix).resolve()
    if not input_solution.exists():
        raise FileNotFoundError(f"Missing input solution file: {input_solution}")

    safe_id = _safe_id_from_matrix(input_matrix)
    output_name = f"solutions_{safe_id}.csv"

    solutions_dir = _resolve_solutions_dir(args.solutions_dir)
    solutions_dir.mkdir(parents=True, exist_ok=True)
    if _is_ephemeral_solutions_dir(solutions_dir):
        print(
            "QSPLIT WARNING solutions_dir points to a temporary StreamFlow directory. "
            "Use an absolute path in cwl/config.yml (solutions_store_dir).",
            flush=True,
        )
    persisted_target = solutions_dir / output_name
    shutil.copy2(input_solution, persisted_target)
    print(f"QSPLIT PERSIST solution={persisted_target}", flush=True)

    output_solution = Path(args.output_solution).resolve()
    shutil.copy2(persisted_target, output_solution)


if __name__ == "__main__":
    main()
