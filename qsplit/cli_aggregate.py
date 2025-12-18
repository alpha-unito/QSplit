import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .io_utils import load_qubo, save_qubo
from .qubo import QUBO


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate QSplit outputs (no D-Wave dependency)")
    parser.add_argument("--input-qubo", required=True, help="initial_qubo.pkl")
    parser.add_argument("--tree-file", required=True, help="tree.json")
    parser.add_argument("--output-qubo", default="aggregate.pkl")
    parser.add_argument(
        "--solved-list",
        default="",
        help="Comma-separated list of solved sub-QUBO pkls (can be empty if solve skipped)",
    )
    args = parser.parse_args()

    full_qubo = load_qubo(Path(args.input_qubo))
    if not isinstance(full_qubo, QUBO):
        raise TypeError(f"Expected QUBO in {args.input_qubo}, got {type(full_qubo)}")

    tree = json.loads(Path(args.tree_file).read_text(encoding="utf-8"))
    nodes: Dict[str, Dict] = tree.get("nodes", {})

    mat = full_qubo.mat.copy()
    mat[np.abs(mat) < 1e-12] = 0.0
    np.savetxt("aggregate.csv", mat, delimiter=",", fmt="%.12g")

    solved_paths: List[Path] = [Path(p) for p in (args.solved_list.split(",") if args.solved_list else []) if p.strip()]

    lines: List[str] = ["node_id,backend,solution_index,bitstring,energy"]

    for sp in solved_paths:
        qubo = load_qubo(sp)
        if not isinstance(qubo, QUBO):
            continue
        node_id = getattr(qubo, "node_id", sp.stem)
        backend = str(nodes.get(node_id, {}).get("backend", ""))

        df = getattr(qubo, "solutions", None)
        if df is None or getattr(df, "empty", True):
            continue

        bit_cols = [c for c in df.columns if c != "energy"]

        def _k(x):
            try:
                return int(x)
            except Exception:
                return 10**18

        bit_cols = sorted(bit_cols, key=_k)

        for i, row in df.reset_index(drop=True).iterrows():
            try:
                bits = "".join(str(int(row[c])) for c in bit_cols)
            except Exception:
                bits = ""
            try:
                energy = float(row["energy"])
            except Exception:
                energy = float("nan")
            lines.append(f"{node_id},{backend},{i},{bits},{energy:.12g}")

    Path("aggregate.solutions.csv").write_text("\n".join(lines), encoding="utf-8")

    save_qubo(Path(args.output_qubo), full_qubo)


if __name__ == "__main__":
    main()