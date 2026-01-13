import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
from .io_utils import load_qubo, save_qubo
from qsplit.halting_heuristic.stop import is_empty, is_sparse
from qsplit.qubo import QUBO
from qsplit.splitting.split_recursive import split_problem
from qsplit.halting_heuristic.stop import is_empty, is_sparse
import os

_backend_counts: Dict[str, int] = {}


def _build_qubo_from_matrix(matrix_path: str) -> QUBO:
    mat = np.loadtxt(matrix_path, delimiter=",")
    n = mat.shape[0]
    qubo = QUBO(mat=mat, rows_idx=np.arange(n), cols_idx=np.arange(n))
    save_qubo("initial_qubo.pkl", qubo)
    return qubo


def _parse_backends(backends_csv: str) -> List[str]:
    xs = [b.strip() for b in (backends_csv or "").split(",") if b.strip()]
    return xs or ["dwave"]


def _parse_backend_cut_dims(spec: str) -> Dict[str, int]:
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


def _eligible_backends(size: int, backends: List[str], backend_cut_dims: Dict[str, int]) -> List[str]:
    elig = []
    for b in backends:
        lim = backend_cut_dims.get(b)
        if lim is None:
            continue
        if size <= max(int(lim), 1):
            elig.append(b)
    return elig


def _sparse_variable_count(qubo: QUBO) -> int:
    rows_found, cols_found = np.nonzero(qubo.mat)
    variables_in_rows = qubo.rows_idx[rows_found]
    variables_in_cols = qubo.cols_idx[cols_found]
    unique_vars = np.unique(np.concatenate([variables_in_rows, variables_in_cols]))
    return len(unique_vars)


def _choose_backend_weighted_rr(eligible: List[str], backend_cut_dims: Dict[str, int]) -> str:
    def weight(b: str) -> int:
        return max(int(backend_cut_dims.get(b, 1)), 1)

    def score(b: str) -> float:
        return _backend_counts.get(b, 0) / float(weight(b))

    chosen = min(eligible, key=lambda b: (score(b), eligible.index(b)))
    _backend_counts[chosen] = _backend_counts.get(chosen, 0) + 1
    return chosen


def _recursively_split(
    qubo: QUBO,
    node_id: str,
    out_dir: Path,
    nodes: Dict[str, Dict],
    backends: List[str],
    backend_cut_dims: Dict[str, int],
    cut_dim: int,
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

    sparse = is_sparse(qubo)
    size = int(getattr(qubo, "problem_size", qubo.mat.shape[0]))
    if is_empty(qubo):
        nodes[node_id]["backend"] = "dummy"
        bdir = out_dir / "dummy"
        bdir.mkdir(parents=True, exist_ok=True)
        save_qubo(bdir / f"{node_id}.pkl", qubo)
        return

    sparse = is_sparse(qubo)
    effective_size = _sparse_variable_count(qubo) if sparse else size
    eligible = _eligible_backends(effective_size, backends, backend_cut_dims)

    if ((size <= cut_dim) or sparse) and eligible:
        chosen = _choose_backend_weighted_rr(eligible, backend_cut_dims)
        nodes[node_id]["backend"] = chosen

        bdir = out_dir / chosen
        bdir.mkdir(parents=True, exist_ok=True)
        save_qubo(bdir / f"{node_id}.pkl", qubo)
        return

    for idx, sub in enumerate(split_problem(qubo)):
        child_id = f"{node_id}_{idx}"
        nodes[node_id]["children"].append(child_id)
        _recursively_split(sub, child_id, out_dir, nodes, backends, backend_cut_dims, cut_dim)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-qubo")
    p.add_argument("--input-matrix")
    p.add_argument("--cut-dim", type=int, required=True)
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--approach", default="dr")
    p.add_argument("--tree-file", default="tree.json")
    p.add_argument("--out-dir", default="subproblems")
    p.add_argument("--backends", default="dwave")
    p.add_argument("--backend-cut-dims", default="")
    p.add_argument("--backend-file", default="backends.json")
    args = p.parse_args()

    os.environ["CUT_DIM"] = str(args.cut_dim)

    if bool(args.input_qubo) == bool(args.input_matrix):
        raise SystemExit("Specify exactly one of --input-qubo or --input-matrix")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUT_DIM"] = str(args.cut_dim)

    if args.input_matrix:
        full = _build_qubo_from_matrix(args.input_matrix)
    else:
        full = load_qubo(args.input_qubo)
        save_qubo("initial_qubo.pkl", full)

    backends = _parse_backends(args.backends)
    backend_cut_dims = _parse_backend_cut_dims(args.backend_cut_dims)

    for b in backends:
        if b not in backend_cut_dims:
            backend_cut_dims[b] = int(args.cut_dim)

    global _backend_counts
    _backend_counts = {b: 0 for b in backends}

    nodes: Dict[str, Dict] = {}
    _recursively_split(full, "root", out_dir, nodes, backends, backend_cut_dims, args.cut_dim)

    Path(args.tree_file).write_text(json.dumps({"root": "root", "nodes": nodes}, indent=2), encoding="utf-8")

    Path(args.backend_file).write_text("[]", encoding="utf-8")


if __name__ == "__main__":
    main()
