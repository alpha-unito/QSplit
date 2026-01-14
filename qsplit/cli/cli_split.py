import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
from .io_utils import save_qubo
from qsplit.halting_heuristic.stop import is_empty, vars_count
from qsplit.qubo import QUBO
from qsplit.splitting.split_recursive import split_problem
import os

backend_counts: Dict[str, int] = {}


def build_qubo_from_matrix(matrix_path: str) -> QUBO:
    mat = np.loadtxt(matrix_path, delimiter=",")
    n = mat.shape[0]
    qubo = QUBO(mat=mat, rows_idx=np.arange(n), cols_idx=np.arange(n))
    save_qubo("initial_qubo.pkl", qubo)
    return qubo


def parse_backend_cut_dims(spec: str) -> Dict[str, int]:
    res: Dict[str, int] = {}
    if not spec:
        return res
    for part in spec.split(","):
        part = part.strip()
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        try:
            res[k] = int(v)
        except ValueError:
            pass
    return res


def eligible_backends(size: int, backends: List[str], backend_cut_dims: Dict[str, int]) -> List[str]:
    elig = []
    for b in backends:
        lim = backend_cut_dims.get(b)
        if lim is None:
            continue
        if size <= max(int(lim), 1):
            elig.append(b)
    return elig


def choose_backend_weighted_rr(
    eligible: List[str],
    backend_cut_dims: Dict[str, int],
    demand: int,
) -> str:
    def cap(b: str) -> int:
        return max(int(backend_cut_dims.get(b, 1)), 1)

    def weight(b: str) -> float:
        c = cap(b)
        margin = max(c - int(demand), 0)
        fit = 1.0 / (1.0 + float(margin))
        return float(c) * fit

    def score(b: str) -> float:
        return backend_counts.get(b, 0) / float(weight(b))

    chosen = min(eligible, key=lambda b: (score(b), eligible.index(b)))
    backend_counts[chosen] = backend_counts.get(chosen, 0) + 1
    return chosen


def recursively_split(
    qubo: QUBO,
    node_id: str,
    out_dir: Path,
    nodes: Dict[str, Dict],
    backends: List[str],
    backend_cut_dims: Dict[str, int],
) -> None:
    qubo.node_id = node_id

    nodes[node_id] = {
        "rows_idx": qubo.rows_idx.astype(int).tolist(),
        "cols_idx": qubo.cols_idx.astype(int).tolist(),
        "offset": float(getattr(qubo, "offset", 0.0)),
        "children": [],
    }

    if is_empty(qubo):
        nodes[node_id]["backend"] = "dummy"
        bdir = out_dir / "dummy"
        bdir.mkdir(parents=True, exist_ok=True)
        save_qubo(bdir / f"{node_id}.pkl", qubo)
        return

    size = qubo.problem_size
    eligible = eligible_backends(size, backends, backend_cut_dims)

    if eligible:
        demand = vars_count(qubo)
        chosen = choose_backend_weighted_rr(eligible, backend_cut_dims, demand)
        nodes[node_id]["backend"] = chosen

        bdir = out_dir / chosen
        bdir.mkdir(parents=True, exist_ok=True)
        save_qubo(bdir / f"{node_id}.pkl", qubo)
        return

    for idx, sub in enumerate(split_problem(qubo)):
        child_id = f"{node_id}_{idx}"
        nodes[node_id]["children"].append(child_id)
        recursively_split(sub, child_id, out_dir, nodes, backends, backend_cut_dims)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-matrix", required=True)
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--approach", default="dr")
    p.add_argument("--out-dir", default="subproblems")
    p.add_argument("--backend-cut-dims", default="")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    full = build_qubo_from_matrix(args.input_matrix)

    backend_cut_dims = parse_backend_cut_dims(args.backend_cut_dims)

    global backend_counts
    backend_counts = {b: 0 for b in backend_cut_dims.keys()}

    nodes: Dict[str, Dict] = {}
    recursively_split(full, "root", out_dir, nodes, backend_cut_dims.keys(), backend_cut_dims)

    Path("tree.json").write_text(json.dumps({"root": "root", "nodes": nodes}, indent=2), encoding="utf-8")
    Path("backends.json").write_text("[]", encoding="utf-8")


if __name__ == "__main__":
    main()
