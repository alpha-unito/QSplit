import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .io_utils import load_qubo, save_qubo
from .qubo import QUBO
from .split import split_problem


def _build_qubo_from_matrix(matrix_path: str) -> QUBO:
    mat = np.loadtxt(matrix_path, delimiter=",")
    n_rows, n_cols = mat.shape
    rows_idx = np.arange(n_rows)
    cols_idx = np.arange(n_cols)
    qubo = QUBO(mat=mat, rows_idx=rows_idx, cols_idx=cols_idx)
    save_qubo("initial_qubo.pkl", qubo)
    return qubo


def print_tree(node_id, nodes, prefix=""):
    children = nodes[node_id].get("children", [])
    backend = nodes[node_id].get("backend")
    backend_label = f" backend={backend}" if backend else ""
    label = f"{node_id}{backend_label} (rows={nodes[node_id]['rows_idx']}, cols={nodes[node_id]['cols_idx']})"
    print(prefix + label)
    for i, child in enumerate(children):
        last = "└─ " if i == len(children) - 1 else "├─ "
        extension = "   " if i == len(children) - 1 else "│  "
        print_tree(child, nodes, prefix + last)
        prefix = prefix[:-3] + extension if prefix.endswith(("├─ ", "└─ ")) else prefix + extension


def _parse_backends(backends_csv: str) -> List[str]:
    backends = [b.strip() for b in (backends_csv or "").split(",") if b.strip()]
    return backends or ["dwave"]


def _parse_backend_cut_dims(spec: str) -> Dict[str, int]:
    """
    Parse a CSV spec like 'dwave:64,ibm:20,iqm:32' into a dict.
    Invalid entries are ignored.
    """
    res: Dict[str, int] = {}
    if not spec:
        return res
    for part in spec.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        key, val = part.split(":", 1)
        key = key.strip()
        val = val.strip()
        if not key or not val:
            continue
        try:
            res[key] = int(val)
        except ValueError:
            continue
    return res


def _choose_backend_for_size(
    size: int,
    backends: List[str],
    backend_cut_dims: Dict[str, int],
    default_cut_dim: int,
) -> Optional[str]:
    """
    Select the first backend (in the given order) that can accept a leaf of this size.
    The 'backend_cut_dims' map defines the maximum leaf size for each backend.
    If a backend is missing from the map, 'default_cut_dim' is used.
    """
    for b in backends:
        threshold = backend_cut_dims.get(b, default_cut_dim)
        if size <= threshold:
            return b
    return None


def _recursively_split(
    qubo: QUBO,
    prefix: str,
    out_dir: Path,
    nodes: Dict[str, Dict],
    leaves: List[Path],
    leaf_backends: List[str],
    backends: List[str],
    backend_cut_dims: Dict[str, int],
    default_cut_dim: int,
) -> None:
    qubo.node_id = prefix

    nodes[prefix] = {
        "rows_idx": qubo.rows_idx.astype(int).tolist(),
        "cols_idx": qubo.cols_idx.astype(int).tolist(),
        "offset": float(qubo.offset),
        "mat": qubo.mat.astype(float).tolist(),
        "children": [],
    }

    # Dynamic stop rule: if this node fits the max size of one of the
    # configured backends, we stop here and assign that backend.
    chosen = _choose_backend_for_size(
        qubo.problem_size, backends, backend_cut_dims, default_cut_dim
    )

    if chosen is not None:
        out_path = out_dir / f"{prefix}.pkl"
        leaves.append(out_path)
        leaf_backends.append(chosen)
        nodes[prefix]["backend"] = chosen
        save_qubo(out_path, qubo)
        return

    # Otherwise keep splitting until we can assign a backend
    for idx, sub in enumerate(split_problem(qubo)):
        child_prefix = f"{prefix}_{idx}"
        nodes[prefix]["children"].append(child_prefix)
        _recursively_split(
            sub,
            child_prefix,
            out_dir,
            nodes,
            leaves,
            leaf_backends,
            backends,
            backend_cut_dims,
            default_cut_dim,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Split QUBO into subproblems")
    parser.add_argument("--input-qubo", help="Input full QUBO (.pkl)")
    parser.add_argument("--input-matrix", help="Input matrix file (.csv) to build the full QUBO")
    parser.add_argument("--cut-dim", type=int, required=True, help="Default max leaf size (fallback)")
    parser.add_argument("--approach", default="dr", help="Splitting approach (currently unused)")
    parser.add_argument("--adaptive", action="store_true", help="Adaptive splitting (currently unused)")
    parser.add_argument("--out-dir", required=True, help="Output directory for sub-QUBOs")
    parser.add_argument("--tree-file", default="tree.json", help="Output metadata file describing the split tree")

    # Multi-target / routing inputs
    parser.add_argument(
        "--backends",
        default="dwave",
        help=(
            "Comma-separated backend labels, ordered by preference for assignment. "
            "Example: 'ibm,iqm,dwave'"
        ),
    )
    parser.add_argument(
        "--backend-cut-dims",
        default="",
        help=(
            "Comma-separated max leaf sizes per backend. "
            "Example: 'ibm:20,iqm:32,dwave:64'. "
            "Missing entries fall back to --cut-dim."
        ),
    )
    parser.add_argument(
        "--backend-file",
        default="backends.json",
        help="Output JSON file containing backend labels aligned with generated leaf sub-QUBOs",
    )

    args = parser.parse_args()

    if bool(args.input_qubo) == bool(args.input_matrix):
        parser.error("You must specify exactly one of --input-qubo or --input-matrix")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input_matrix:
        full_qubo = _build_qubo_from_matrix(args.input_matrix)
    else:
        full_qubo = load_qubo(args.input_qubo)

    backends = _parse_backends(args.backends)
    backend_cut_dims = _parse_backend_cut_dims(args.backend_cut_dims)

    tree_nodes: Dict[str, Dict] = {}
    leaves: List[Path] = []
    leaf_backends: List[str] = []

    _recursively_split(
        full_qubo,
        "root",
        out_dir,
        tree_nodes,
        leaves,
        leaf_backends,
        backends,
        backend_cut_dims,
        args.cut_dim,
    )

    tree_path = Path(args.tree_file).resolve()
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    with tree_path.open("w", encoding="utf-8") as f:
        json.dump({"root": "root", "nodes": tree_nodes}, f, indent=2)

    backend_path = Path(args.backend_file).resolve()
    backend_path.parent.mkdir(parents=True, exist_ok=True)
    with backend_path.open("w", encoding="utf-8") as f:
        json.dump(leaf_backends, f, indent=2)

    # Optional debug print of the tree with backend annotations
    try:
        with tree_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        print_tree(meta["root"], meta["nodes"])
    except Exception:
        pass

    print("[cli_split] Generated leaf sub-QUBOs:")
    for p, b in zip(leaves, leaf_backends):
        print(f"  - {p}  (backend={b})")


if __name__ == "__main__":
    main()