import argparse
from pathlib import Path
from typing import List

import numpy as np

from .qubo import QUBO
from .split import split_problem
from .io_utils import load_qubo, save_qubo


def recursively_split(
    qubo: QUBO,
    cut_dim: int,
    adaptive_splitting: bool,
    approach: str,
    out_dir: Path,
    prefix: str = "root",
) -> List[Path]:

    problem_size = qubo.mat.shape[0]

    if problem_size <= cut_dim:
        out_path = out_dir / f"{prefix}.pkl"
        save_qubo(out_path, qubo)
        print(f"[cli_split] Saved leaf sub-QUBO: {out_path}")
        return [out_path]

    sub_qubos = split_problem(qubo)

    paths: List[Path] = []
    for i, sub in enumerate(sub_qubos):
        child_prefix = f"{prefix}_{i}"
        paths.extend(
            recursively_split(
                sub,
                cut_dim=cut_dim,
                adaptive_splitting=adaptive_splitting,
                approach=approach,
                out_dir=out_dir,
                prefix=child_prefix,
            )
        )
    return paths


def _build_qubo_from_matrix(matrix_path: str) -> QUBO:

    mat = np.loadtxt(matrix_path, delimiter=",")
    n_rows, n_cols = mat.shape

    rows_idx = np.arange(n_rows)
    cols_idx = np.arange(n_cols)

    qubo = QUBO(mat=mat, rows_idx=rows_idx, cols_idx=cols_idx)

    save_qubo("initial_qubo.pkl", qubo)

    return qubo


def main() -> None:
    parser = argparse.ArgumentParser(description="Split QUBO into subproblems")

    parser.add_argument(
        "--input-qubo",
        help="Input full QUBO (.pkl)",
    )

    parser.add_argument(
        "--input-matrix",
        help="Input matrix file (.csv) to build the full QUBO",
    )

    parser.add_argument(
        "--cut-dim",
        type=int,
        required=True,
        help="Max sub-QUBO size",
    )
    parser.add_argument(
        "--approach",
        default="dr",
        help="Splitting approach (currently unused)",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Adaptive splitting (currently unused)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for sub-QUBOs",
    )

    args = parser.parse_args()

    if bool(args.input_qubo) == bool(args.input_matrix):
        parser.error("You must specify exactly one of --input-qubo or --input-matrix")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input_matrix:
        print(f"[cli_split] Building QUBO from matrix: {args.input_matrix}")
        full_qubo = _build_qubo_from_matrix(args.input_matrix)
    else:
        print(f"[cli_split] Loading QUBO from pickle: {args.input_qubo}")
        full_qubo = load_qubo(args.input_qubo)

    print("[cli_split] INPUT QUBO:")
    print("  shape:", full_qubo.mat.shape)
    print("  rows_idx:", full_qubo.rows_idx)
    print("  cols_idx:", full_qubo.cols_idx)

    sub_paths = recursively_split(
        full_qubo,
        cut_dim=args.cut_dim,
        adaptive_splitting=args.adaptive,
        approach=args.approach,
        out_dir=out_dir,
    )

    print("[cli_split] Generated sub-QUBOs:")
    for p in sub_paths:
        print("  -", p)

if __name__ == "__main__":
    main()