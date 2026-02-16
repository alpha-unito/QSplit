import argparse
import json
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
    out_dir: Path,
    solved_dir: Path,
    nodes: Dict[str, Dict],
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
        qubo.solutions = dummy_solve(qubo)
        qubo.backend = "dummy"
        save_qubo(solved_dir / f"{node_id}.pkl", qubo)
        return

    if (int(qubo.problem_size) <= cut_dim) or is_sparse(qubo, cut_dim):
        save_qubo(out_dir / f"{node_id}.pkl", qubo)
        return

    for idx, sub in enumerate(split_problem(qubo)):
        child_id = f"{node_id}_{idx}"
        nodes[node_id]["children"].append(child_id)
        recursively_split(sub, child_id, out_dir, solved_dir, nodes, cut_dim)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-matrix", required=True)
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--approach", default="dr")
    p.add_argument("--out-dir", default="subproblems")
    p.add_argument("--cut-dim", type=int, default=16)
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    solved_dir = Path("solved_dummy").resolve()
    solved_dir.mkdir(parents=True, exist_ok=True)

    full = build_qubo_from_matrix(args.input_matrix)

    cut_dim = int(args.cut_dim)
    if cut_dim <= 0:
        raise ValueError(f"cut_dim must be positive, got {cut_dim}")

    nodes: Dict[str, Dict] = {}
    recursively_split(full, "root", out_dir, solved_dir, nodes, cut_dim)

    Path("tree.json").write_text(json.dumps({"root": "root", "nodes": nodes}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
