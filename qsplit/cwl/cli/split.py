import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict

from qsplit.adapters.dummy import solve as dummy_solve
from qsplit.cwl.cli.utils import build_qubo_from_matrix, save_qubo
from qsplit.halting_heuristic.stop import is_empty, is_sparse
from qsplit.qubo import QUBO
from qsplit.splitting.split_recursive import split_problem


def recursively_split(
    qubo: QUBO,
    node_id: str,
    instance_id: str,
    out_dir: Path,
    solved_dir: Path,
    nodes: Dict[str, Dict],
    cut_dim: int,
    enable_sparse_check: bool,
) -> None:
    qubo.node_id = node_id
    qubo.instance_id = instance_id

    nodes[node_id] = {
        "rows_idx": qubo.rows_idx.astype(int).tolist(),
        "cols_idx": qubo.cols_idx.astype(int).tolist(),
        "offset": float(getattr(qubo, "offset", 0.0)),
        "children": [],
    }

    if is_empty(qubo):
        qubo.solutions = dummy_solve(qubo)
        qubo.backend = "dummy"
        save_qubo(solved_dir / f"{node_id}.pkl", qubo)
        return

    if (int(qubo.problem_size) <= cut_dim) or (enable_sparse_check and is_sparse(qubo, cut_dim)):
        save_qubo(out_dir / f"{node_id}.pkl", qubo)
        return

    for idx, sub in enumerate(split_problem(qubo)):
        child_id = f"{node_id}_{idx}"
        nodes[node_id]["children"].append(child_id)
        recursively_split(
            sub,
            child_id,
            instance_id,
            out_dir,
            solved_dir,
            nodes,
            cut_dim,
            enable_sparse_check,
        )


def _parse_count(raw: str) -> int | None:
    value = (raw or "").strip().lower()
    if value == "auto":
        return None
    if not value:
        return 1
    try:
        n = int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid count '{raw}'. Use integer >= 0 or 'auto'.") from exc
    if n < 0:
        raise ValueError(f"Invalid count '{raw}'. Must be >= 0.")
    return n


def _take(items: list[Path], n: int) -> list[Path]:
    n = min(n, len(items))
    out = items[:n]
    del items[:n]
    return out


def _allocate_subproblems(
    sub_qubos: list[Path],
    *,
    enable_iqm: bool,
    enable_quantinuum_h2: bool,
    enable_quantinuum_h2e: bool,
    iqm_real_jobs: str,
    quantinuum_h2_real_jobs: str,
    quantinuum_h2e_real_jobs: str,
) -> dict[str, list[Path]]:
    remaining = list(sub_qubos)
    assignments: dict[str, list[Path]] = {
        "iqm": [],
        "quantinuum_h2": [],
        "quantinuum_h2e": [],
        "parallel": [],
    }
    enabled = {
        "iqm": enable_iqm,
        "quantinuum_h2": enable_quantinuum_h2,
        "quantinuum_h2e": enable_quantinuum_h2e,
    }
    raw_counts = {
        "iqm": iqm_real_jobs,
        "quantinuum_h2": quantinuum_h2_real_jobs,
        "quantinuum_h2e": quantinuum_h2e_real_jobs,
    }

    auto_backends: list[str] = []
    for backend in ("iqm", "quantinuum_h2", "quantinuum_h2e"):
        if not enabled[backend]:
            continue
        parsed = _parse_count(raw_counts[backend])
        if parsed is None:
            auto_backends.append(backend)
            continue
        assignments[backend] = _take(remaining, parsed)

    if auto_backends:
        lanes = ["parallel"] + auto_backends
        total = len(remaining)
        base = total // len(lanes)
        rem = total % len(lanes)
        cursor = 0
        for idx, lane in enumerate(lanes):
            take = base + (1 if idx < rem else 0)
            assignments[lane] = remaining[cursor : cursor + take]
            cursor += take
    else:
        assignments["parallel"] = remaining

    return assignments


def _materialize(files: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.pkl"):
        stale.unlink()
    for idx, src in enumerate(files):
        dst = out_dir / f"{idx:06d}_{src.name}"
        if dst.exists():
            dst.unlink()
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)


def _instance_id_from_matrix_path(matrix_path: str) -> str:
    stem = Path(matrix_path).stem
    match = re.match(r"^\d{6}_(.+)$", stem)
    if match:
        candidate = match.group(1)
        nested = re.match(r"^\d{6}_(.+)$", candidate)
        if nested:
            candidate = nested.group(1)
        stem = candidate
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return safe or "instance_unknown"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-matrix", required=True)
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--approach", default="dr")
    p.add_argument("--out-dir", default="subproblems")
    p.add_argument("--cut-dim", type=int, default=16)
    p.add_argument("--enable-sparse-check", action="store_true")
    p.add_argument("--enable-iqm", action="store_true")
    p.add_argument("--enable-quantinuum-h2", action="store_true")
    p.add_argument("--enable-quantinuum-h2e", action="store_true")
    p.add_argument("--iqm-real-jobs", default="1")
    p.add_argument("--quantinuum-h2-real-jobs", default="1")
    p.add_argument("--quantinuum-h2e-real-jobs", default="1")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    solved_dir = Path("solved_dummy").resolve()
    solved_dir.mkdir(parents=True, exist_ok=True)

    instance_id = _instance_id_from_matrix_path(args.input_matrix)
    full = build_qubo_from_matrix(args.input_matrix)
    # Keep the full problem metadata aligned with subproblems so downstream
    # aggregation can validate instance consistency.
    full.instance_id = instance_id
    full.node_id = "root"
    save_qubo("initial_qubo.pkl", full)

    cut_dim = int(args.cut_dim)
    if cut_dim <= 0:
        raise ValueError(f"cut_dim must be positive, got {cut_dim}")

    nodes: Dict[str, Dict] = {}
    recursively_split(
        full,
        "root",
        instance_id,
        out_dir,
        solved_dir,
        nodes,
        cut_dim,
        bool(args.enable_sparse_check),
    )

    assignments = _allocate_subproblems(
        sorted(out_dir.glob("*.pkl")),
        enable_iqm=bool(args.enable_iqm),
        enable_quantinuum_h2=bool(args.enable_quantinuum_h2),
        enable_quantinuum_h2e=bool(args.enable_quantinuum_h2e),
        iqm_real_jobs=str(args.iqm_real_jobs),
        quantinuum_h2_real_jobs=str(args.quantinuum_h2_real_jobs),
        quantinuum_h2e_real_jobs=str(args.quantinuum_h2e_real_jobs),
    )
    planned_root = Path("planned").resolve()
    _materialize(assignments["iqm"], planned_root / "iqm")
    _materialize(assignments["quantinuum_h2"], planned_root / "quantinuum_h2")
    _materialize(assignments["quantinuum_h2e"], planned_root / "quantinuum_h2e")
    _materialize(assignments["parallel"], planned_root / "parallel")

    Path("tree.json").write_text(json.dumps({"root": "root", "nodes": nodes}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
