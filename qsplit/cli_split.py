import argparse
import json
from pathlib import Path
from typing import Dict, List

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
    label = f"{node_id} (rows={nodes[node_id]['rows_idx']}, cols={nodes[node_id]['cols_idx']})"
    print(prefix + label)
    for i, child in enumerate(children):
        last = "└─ " if i == len(children) - 1 else "├─ "
        extension = "   " if i == len(children) - 1 else "│  "
        print_tree(child, nodes, prefix + last)
        prefix = prefix[:-3] + extension if prefix.endswith(("├─ ", "└─ ")) else prefix + extension


def _recursively_split(
    qubo: QUBO,
    prefix: str,
    cut_dim: int,
    out_dir: Path,
    nodes: Dict[str, Dict[str, List[str]]],
    leaves: List[Path],
) -> None:
    qubo.node_id = prefix

    nodes[prefix] = {
        "rows_idx": qubo.rows_idx.astype(int).tolist(),
        "cols_idx": qubo.cols_idx.astype(int).tolist(),
        "offset": float(qubo.offset),
        "mat": qubo.mat.astype(float).tolist(),
        "children": [],
    }

    if qubo.problem_size <= cut_dim:
        out_path = out_dir / f"{prefix}.pkl"
        leaves.append(out_path)
        save_qubo(out_path, qubo)
        return

    for idx, sub in enumerate(split_problem(qubo)):
        child_prefix = f"{prefix}_{idx}"
        nodes[prefix]["children"].append(child_prefix)
        _recursively_split(sub, child_prefix, cut_dim, out_dir, nodes, leaves)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split QUBO into subproblems")
    parser.add_argument("--input-qubo", help="Input full QUBO (.pkl)")
    parser.add_argument("--input-matrix", help="Input matrix file (.csv) to build the full QUBO")
    parser.add_argument("--cut-dim", type=int, required=True, help="Max sub-QUBO size")
    parser.add_argument("--approach", default="dr", help="Splitting approach (currently unused)")
    parser.add_argument("--adaptive", action="store_true", help="Adaptive splitting (currently unused)")
    parser.add_argument("--out-dir", required=True, help="Output directory for sub-QUBOs")
    parser.add_argument("--tree-file", default="tree.json", help="Output metadata file describing the split tree")
    args = parser.parse_args()

    if bool(args.input_qubo) == bool(args.input_matrix):
        parser.error("You must specify exactly one of --input-qubo or --input-matrix")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input_matrix:
        full_qubo = _build_qubo_from_matrix(args.input_matrix)
    else:
        full_qubo = load_qubo(args.input_qubo)

    tree_nodes: Dict[str, Dict[str, List[str]]] = {}
    leaves: List[Path] = []
    _recursively_split(full_qubo, "root", args.cut_dim, out_dir, tree_nodes, leaves)

    tree_path = Path(args.tree_file).resolve()
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    with tree_path.open("w", encoding="utf-8") as f:
        json.dump({"root": "root", "nodes": tree_nodes}, f, indent=2)

    # tree_path = Path("tree.json")  # oppure /tmp/.../tree.json
    # with tree_path.open("r", encoding="utf-8") as fh:
    #     meta = json.load(fh)

    # print_tree(meta["root"], meta["nodes"]) 

    print("[cli_split] Generated leaf sub-QUBOs:")
    for p in leaves:
        print("  -", p)


if __name__ == "__main__":
    main()
