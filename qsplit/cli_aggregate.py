import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from . import aggregate as aggregate_module
from .aggregate import aggregate_solutions
from .adapters.all_zero import solve as zero_solver
from .io_utils import load_qubo, save_qubo
from .qubo import QUBO

KeyType = Tuple[Tuple[int, ...], Tuple[int, ...]]


def _solution_key(qubo: QUBO) -> KeyType:
    rows = tuple(int(v) for v in qubo.rows_idx.tolist())
    cols = tuple(int(v) for v in qubo.cols_idx.tolist())
    return rows, cols


def _normalize_solution_columns(qubo: QUBO) -> None:
    if qubo.solutions is None:
        return
    numeric_cols = [c for c in qubo.solutions.columns if c != "energy"]
    target_order = [int(v) for v in qubo.cols_idx.tolist()]
    if len(numeric_cols) != len(target_order):
        raise ValueError(
            f"Inconsistent solution width for rows {qubo.rows_idx} / cols {qubo.cols_idx}: "
            f"{len(numeric_cols)} values vs {len(target_order)} expected."
        )
    rename = {old: target_order[idx] for idx, old in enumerate(numeric_cols)}
    qubo.solutions = qubo.solutions.rename(columns=rename)
    qubo.solutions = qubo.solutions[target_order + ["energy"]]


def _print_solutions(node_id: str, qubo: QUBO) -> None:
    if qubo.solutions is None or qubo.solutions.empty:
        print(f"[cli_aggregate] Node {node_id}: no solutions")
        return

    print(
        f"[cli_aggregate] Node {node_id} "
        f"(rows={list(qubo.rows_idx)}, cols={list(qubo.cols_idx)}) solutions:"
    )
    for idx, row in qubo.solutions.reset_index(drop=True).iterrows():
        assignments = {
            int(col): int(row[col]) for col in qubo.solutions.columns if col != "energy"
        }
        print(f"  - sol #{idx + 1} energy={row['energy']}, assignments={assignments}")


def _load_solved_qubos(paths: Iterable[Path]) -> Dict[str, QUBO]:
    solved: Dict[str, QUBO] = {}
    for path in paths:
        qubo = load_qubo(path)
        if not isinstance(qubo, QUBO):
            raise TypeError(f"{path} does not contain a QUBO object (found {type(qubo)})")
        if qubo.solutions is None or qubo.solutions.empty:
            raise ValueError(f"{path} has no solutions to aggregate")
        _normalize_solution_columns(qubo)
        node_id = getattr(qubo, "node_id", None)
        if not node_id:
            raise ValueError(f"{path} is missing node_id metadata. Re-run split to include node identifiers.")
        if node_id in solved:
            raise ValueError(f"Duplicate solved QUBO for node '{node_id}'")
        # _print_solutions(node_id, qubo)
        solved[node_id] = qubo
    if not solved:
        raise ValueError("No solved sub-QUBOs provided")
    return solved


def _build_qubo_from_meta(meta: Dict[str, object]) -> QUBO:
    qubo = QUBO.__new__(QUBO)
    qubo.mat = np.array(meta["mat"], dtype=np.float64)
    qubo.rows_idx = np.array(meta["rows_idx"], dtype=int)
    qubo.cols_idx = np.array(meta["cols_idx"], dtype=int)
    qubo.offset = float(meta.get("offset", 0.0))
    qubo.problem_size = qubo.mat.shape[0]
    qubo.solutions = None
    return qubo


def _load_tree(tree_path: Path) -> tuple[str, Dict[str, Dict[str, object]], set[str]]:
    with tree_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if "root" not in raw or "nodes" not in raw:
        raise ValueError(f"Invalid tree metadata: missing 'root' or 'nodes' in {tree_path}")

    nodes: Dict[str, Dict[str, object]] = {}
    leaf_ids: set[str] = set()
    for node_id, meta in raw["nodes"].items():
        nodes[node_id] = {
            "rows_idx": meta["rows_idx"],
            "cols_idx": meta["cols_idx"],
            "offset": meta.get("offset", 0.0),
            "mat": meta["mat"],
            "children": meta.get("children", []),
        }
        if not nodes[node_id]["children"]:
            leaf_ids.add(node_id)
    root_id = raw["root"]
    if root_id not in nodes:
        raise ValueError(f"Root node '{root_id}' not found in metadata file {tree_path}")
    return root_id, nodes, leaf_ids


def _patched_local_search(df: pd.DataFrame, qubo: QUBO) -> pd.DataFrame:
    lookup = {int(label): pos for pos, label in enumerate(qubo.cols_idx.tolist())}
    col_labels = [int(c) for c in df.columns if c != "energy"]

    for i, row in df.iterrows():
        valid = [(label, lookup[label]) for label in col_labels if label in lookup]
        if not valid:
            df.loc[i, "energy"] = np.nan
            continue

        labels, positions = zip(*valid)
        label_list = list(labels)
        position_list = list(positions)

        assignments = row[label_list].to_numpy(dtype=float)
        sub_mat = qubo.mat[np.ix_(position_list, position_list)]

        if np.any(np.isnan(assignments)):
            nan_idx = np.where(np.isnan(assignments))[0]
            nan_labels = np.array([label_list[j] for j in nan_idx], dtype=int)
            nan_mat = sub_mat[np.ix_(nan_idx, nan_idx)]
            partial_qubo = QUBO(nan_mat, cols_idx=nan_labels, rows_idx=nan_labels)
            nans_sol = zero_solver(partial_qubo)
            best = nans_sol.nsmallest(1, columns="energy").iloc[0]
            df.loc[i, nan_labels] = best[nan_labels]
            assignments = df.loc[i, label_list].to_numpy(dtype=float)

        df.loc[i, "energy"] = assignments @ sub_mat @ assignments

    return df


aggregate_module.local_search = _patched_local_search


def _rebuild_from_tree(
    node_id: str,
    tree: Dict[str, Dict[str, object]],
    solved: Dict[str, QUBO],
    used: set[KeyType],
) -> QUBO:
    if node_id not in tree:
        raise ValueError(f"Node '{node_id}' not present in tree metadata")

    node_meta = tree[node_id]
    qubo = _build_qubo_from_meta(node_meta)

    if node_id in solved:
        leaf_qubo = solved[node_id]
        used.add(_solution_key(leaf_qubo))
        return leaf_qubo

    children_ids = node_meta.get("children", [])
    if not children_ids:
        raise ValueError(
            f"No solved sub-QUBO provided for leaf node '{node_id}' "
            f"(rows {qubo.rows_idx}, cols {qubo.cols_idx})"
        )

    children = [_rebuild_from_tree(child_id, tree, solved, used) for child_id in children_ids]
    return aggregate_solutions(children, qubo)


def _parse_solved_list(raw: str) -> list[Path]:
    parts = [s.strip() for s in raw.split(",") if s.strip()]
    if not parts:
        raise ValueError("Empty --solved-list")
    return [Path(p) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate QSplit solutions")
    parser.add_argument("--input-qubo", required=True, help="Full QUBO file (.pkl)")
    parser.add_argument("--solved-list", required=True, help="Comma-separated list of solved sub-QUBO .pkl files")
    parser.add_argument("--tree-file", required=True, help="Tree metadata generated during split")
    parser.add_argument("--output-qubo", required=True, help="Output aggregated QUBO (.pkl)")
    args = parser.parse_args()

    input_qubo_path = Path(args.input_qubo)
    if not input_qubo_path.exists():
        raise FileNotFoundError(f"--input-qubo file not found: {args.input_qubo}")

    solved_paths = _parse_solved_list(args.solved_list)
    solved_map = _load_solved_qubos(solved_paths)
    root_id, tree_nodes, leaf_ids = _load_tree(Path(args.tree_file))

    used_keys: set[KeyType] = set()
    aggregated = _rebuild_from_tree(root_id, tree_nodes, solved_map, used_keys)

    missing_nodes = leaf_ids - set(solved_map.keys())
    if missing_nodes:
        raise ValueError(f"Missing solved sub-QUBOs for node IDs: {sorted(missing_nodes)}")

    out_path = Path(args.output_qubo)
    save_qubo(out_path, aggregated)

    csv_path = out_path.with_suffix(".csv")
    mat = aggregated.mat.copy()
    mat[np.abs(mat) < 1e-9] = 0.0
    np.savetxt(csv_path, mat, delimiter=",", fmt="%.6f")


if __name__ == "__main__":
    main()
