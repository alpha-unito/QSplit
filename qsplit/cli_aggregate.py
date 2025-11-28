import argparse
import pickle
from pathlib import Path

import numpy as np

from .io_utils import load_qubo
from .qubo import QUBO


def load_solution(path: Path) -> dict[int, int]:
    obj = load_qubo(path)

    if isinstance(obj, QUBO):
        sub = obj
    else:
        raise TypeError(f"{path} is not a QUBO object: {type(obj)}")

    if sub.solutions is None or sub.solutions.empty:
        return {}

    best = sub.solutions.iloc[0].values
    mapping: dict[int, int] = {}

    for var, val in zip(sub.cols_idx, best):
        if var != -1:
            mapping[int(var)] = int(val)

    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate QSplit solutions")
    parser.add_argument(
        "--input-qubo",
        required=True,
        help="Full QUBO file (.pkl)",
    )
    parser.add_argument(
        "--solved-list",
        required=True,
        help="Comma-separated list of solved sub-QUBO .pkl files",
    )
    parser.add_argument(
        "--output-qubo",
        required=True,
        help="Output aggregated QUBO (.pkl)",
    )

    args = parser.parse_args()

    full_qubo: QUBO = load_qubo(Path(args.input_qubo))

    global_sol: dict[int, int] = {int(v): 0 for v in full_qubo.cols_idx}

    raw_paths = [s.strip() for s in args.solved_list.split(",") if s.strip()]
    solved_paths = [Path(p) for p in raw_paths]

    print("[cli_aggregate] Solved sub-QUBOs:")
    for p in solved_paths:
        print("  -", p)

    for p in solved_paths:
        sol = load_solution(p)
        for var, val in sol.items():
            global_sol[var] = val

    full_qubo.solutions_dict = global_sol

    out_path = Path(args.output_qubo)
    with out_path.open("wb") as f:
        pickle.dump(full_qubo, f)

    csv_path = out_path.with_suffix(".csv")
    mat = full_qubo.mat.copy()
    mat[np.abs(mat) < 1e-9] = 0.0

    np.savetxt(
        csv_path,
        mat,
        delimiter=",",
        fmt="%.6f"
    )

    print("QUBO aggregated:")
    print("mat shape:", full_qubo.mat.shape)
    print("mat:\n", full_qubo.mat)
    print("rows_idx:", full_qubo.rows_idx)
    print("cols_idx:", full_qubo.cols_idx)
    print("offset:", full_qubo.offset)
    if hasattr(full_qubo, "solutions_dict"):
        print("solutions_dict:", full_qubo.solutions_dict)
    if full_qubo.solutions is not None:
        print("solutions DataFrame:")
        print(full_qubo.solutions)

if __name__ == "__main__":
    main()