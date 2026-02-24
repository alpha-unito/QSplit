import argparse
import os
import re
import shutil
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_solutions_dir(raw: str) -> Path:
    base = Path(raw).expanduser()
    if base.is_absolute():
        return base
    launch_dir = os.getenv("QSPLIT_LAUNCH_DIR", "").strip()
    if launch_dir:
        return Path(launch_dir).expanduser() / base
    return _repo_root() / base


def _safe_id_from_matrix(path: Path) -> str:
    stem = path.stem
    match = re.match(r"^\d+_(.+)$", stem)
    if match:
        return match.group(1)
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Persist a completed instance solution to the shared solutions dir.")
    parser.add_argument("--input-solution", required=True)
    parser.add_argument("--input-matrix", required=True)
    parser.add_argument("--solutions-dir", default="solutions")
    parser.add_argument("--output-solution", default="persisted_solution.csv")
    args = parser.parse_args()

    input_solution = Path(args.input_solution).resolve()
    input_matrix = Path(args.input_matrix).resolve()
    if not input_solution.exists():
        raise FileNotFoundError(f"Missing input solution file: {input_solution}")

    safe_id = _safe_id_from_matrix(input_matrix)
    output_name = f"solutions_{safe_id}.csv"

    solutions_dir = _resolve_solutions_dir(args.solutions_dir)
    solutions_dir.mkdir(parents=True, exist_ok=True)
    persisted_target = solutions_dir / output_name
    shutil.copy2(input_solution, persisted_target)

    output_solution = Path(args.output_solution).resolve()
    shutil.copy2(persisted_target, output_solution)


if __name__ == "__main__":
    main()
