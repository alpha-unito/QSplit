import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .aggregation.aggregate_recursive import aggregate_solutions
from .io_utils import load_qubo, save_qubo
from .qubo import QUBO


def _parse_solved_paths(raw: str) -> List[Path]:
    return [Path(p) for p in (raw.split(",") if raw else []) if p.strip()]


def _load_solved_qubos(paths: Iterable[Path]) -> Tuple[List[Tuple[Path, str, QUBO]], Dict[str, QUBO]]:
    entries: List[Tuple[Path, str, QUBO]] = []
    by_id: Dict[str, QUBO] = {}
    for path in paths:
        try:
            qubo = load_qubo(path)
        except Exception:
            continue
        if not isinstance(qubo, QUBO):
            continue
        node_id = getattr(qubo, "node_id", path.stem)
        entries.append((path, node_id, qubo))
        by_id[node_id] = qubo
    return entries, by_id


def _infer_index_order(qubos: Iterable[QUBO]) -> np.ndarray:
    seen: set[int] = set()
    values: List[int] = []
    for qubo in qubos:
        for idx in np.concatenate([qubo.rows_idx, qubo.cols_idx]):
            val = int(idx)
            if val not in seen:
                seen.add(val)
                values.append(val)
    if not values:
        return np.array([], dtype=int)

    def _sort_key(x: int) -> tuple[bool, int]:
        return (x == -1, x)

    return np.array(sorted(values, key=_sort_key), dtype=int)


def _build_index_maps(rows_idx: np.ndarray, cols_idx: np.ndarray) -> tuple[dict[int, int], dict[int, int]]:
    row_map = {int(v): i for i, v in enumerate(rows_idx)}
    col_map = {int(v): i for i, v in enumerate(cols_idx)}
    return row_map, col_map


def _map_indices(idxs: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    return np.array([mapping[int(v)] for v in idxs], dtype=int)


def _reconstruct_full_qubo(
    nodes: Dict[str, Dict],
    root_id: str,
    solved_qubos: Dict[str, QUBO],
) -> tuple[QUBO, dict[int, int], dict[int, int]]:
    root = nodes.get(root_id, {})
    rows_idx = root.get("rows_idx") or []
    cols_idx = root.get("cols_idx") or []
    offset = float(root.get("offset", 0.0))
    if rows_idx and cols_idx:
        rows = np.array(rows_idx, dtype=int)
        cols = np.array(cols_idx, dtype=int)
    else:
        inferred = _infer_index_order(solved_qubos.values())
        rows = inferred
        cols = inferred.copy()

    if rows.size == 0 or cols.size == 0:
        raise SystemExit("Unable to infer full QUBO indices from tree or sub-QUBOs.")
    if rows.shape[0] != cols.shape[0]:
        raise SystemExit("Mismatched row/column index lengths while reconstructing full QUBO.")

    row_map, col_map = _build_index_maps(rows, cols)
    full_mat = np.zeros((rows.shape[0], cols.shape[0]), dtype=float)
    filled = np.zeros_like(full_mat, dtype=bool)

    for node_id, qubo in solved_qubos.items():
        row_pos = _map_indices(qubo.rows_idx, row_map)
        col_pos = _map_indices(qubo.cols_idx, col_map)
        block = np.ix_(row_pos, col_pos)
        filled_block = filled[block]
        if np.any(filled_block):
            existing = full_mat[block]
            mismatch = filled_block & ~np.isclose(existing, qubo.mat, atol=1e-12, rtol=1e-9)
            if np.any(mismatch):
                raise ValueError(f"Conflicting entries while reconstructing full QUBO (node '{node_id}').")
        full_mat[block] = qubo.mat
        filled[block] = True

    full_qubo = QUBO(full_mat, rows_idx=rows, cols_idx=cols, offset=offset)
    return full_qubo, row_map, col_map


def _build_node_qubo(
    node: Dict,
    full_mat: np.ndarray,
    row_map: Dict[int, int],
    col_map: Dict[int, int],
) -> QUBO | None:
    rows_idx = node.get("rows_idx") or []
    cols_idx = node.get("cols_idx") or []
    if not rows_idx or not cols_idx:
        return None
    rows = np.array(rows_idx, dtype=int)
    cols = np.array(cols_idx, dtype=int)
    row_pos = _map_indices(rows, row_map)
    col_pos = _map_indices(cols, col_map)
    mat = full_mat[np.ix_(row_pos, col_pos)]
    offset = float(node.get("offset", 0.0))
    return QUBO(mat, rows_idx=rows, cols_idx=cols, offset=offset)


def _child_sort_key(child_id: str) -> tuple[int, str]:
    try:
        return int(child_id.rsplit("_", 1)[1]), child_id
    except Exception:
        return 10**9, child_id


def _aggregate_tree_solutions(
    root_id: str,
    nodes: Dict[str, Dict],
    solved_qubos: Dict[str, QUBO],
    full_mat: np.ndarray,
    row_map: Dict[int, int],
    col_map: Dict[int, int],
) -> QUBO | None:
    def _visit(node_id: str) -> QUBO | None:
        node = nodes.get(node_id, {})
        children = node.get("children") or []
        if not children:
            return solved_qubos.get(node_id)
        if len(children) != 3:
            raise ValueError(f"Node '{node_id}' expected 3 children, got {len(children)}.")
        child_qubos = [_visit(cid) for cid in sorted(children, key=_child_sort_key)]
        if any(q is None for q in child_qubos):
            return None
        if any(q.solutions is None or getattr(q.solutions, "empty", True) for q in child_qubos):
            return None
        node_qubo = _build_node_qubo(node, full_mat, row_map, col_map)
        if node_qubo is None:
            return None
        return aggregate_solutions(child_qubos, node_qubo)

    return _visit(root_id)


def _bitstring_from_row(row: pd.Series, ordered_cols: List[int]) -> str:
    bits: List[str] = []
    idx = row.index
    for col in ordered_cols:
        if col in idx:
            val = row[col]
        elif str(col) in idx:
            val = row[str(col)]
        else:
            bits.append("")
            continue
        if isinstance(val, pd.Series):
            if val.empty:
                bits.append("")
                continue
            val = val.iloc[0]
        if pd.isna(val):
            bits.append("")
            continue
        try:
            bits.append(str(int(val)))
        except Exception:
            bits.append("")
    return "".join(bits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate QSplit outputs (no D-Wave dependency)")
    parser.add_argument("--input-qubo", help="initial_qubo.pkl (optional fallback)")
    parser.add_argument("--tree-file", required=True, help="tree.json")
    parser.add_argument("--output-qubo", default="aggregate.pkl")
    parser.add_argument(
        "--solved-list",
        default="",
        help="Comma-separated list of solved sub-QUBO pkls (can be empty if solve skipped)",
    )
    args = parser.parse_args()

    tree = json.loads(Path(args.tree_file).read_text(encoding="utf-8"))
    nodes: Dict[str, Dict] = tree.get("nodes", {})
    root_id = tree.get("root", "root")

    solved_paths = _parse_solved_paths(args.solved_list)
    solved_entries, solved_by_id = _load_solved_qubos(solved_paths)

    if solved_by_id:
        full_qubo, row_map, col_map = _reconstruct_full_qubo(nodes, root_id, solved_by_id)
    elif args.input_qubo:
        full_qubo = load_qubo(Path(args.input_qubo))
        if not isinstance(full_qubo, QUBO):
            raise TypeError(f"Expected QUBO in {args.input_qubo}, got {type(full_qubo)}")
        row_map, col_map = _build_index_maps(full_qubo.rows_idx, full_qubo.cols_idx)
    else:
        raise SystemExit("No sub-QUBOs provided; cannot reconstruct the full matrix.")

    mat = full_qubo.mat.copy()
    mat[np.abs(mat) < 1e-12] = 0.0
    np.savetxt("aggregate.csv", mat, delimiter=",", fmt="%.12g")

    lines: List[str] = ["node_id,backend,solution_index,bitstring,energy"]

    for _, node_id, qubo in solved_entries:
        backend = str(nodes.get(node_id, {}).get("backend", ""))

        df = getattr(qubo, "solutions", None)
        if df is None or getattr(df, "empty", True):
            continue

        bit_cols = [int(v) for v in qubo.cols_idx]

        for i, row in df.reset_index(drop=True).iterrows():
            bits = _bitstring_from_row(row, bit_cols)
            try:
                energy = float(row["energy"])
            except Exception:
                energy = float("nan")
            lines.append(f"{node_id},{backend},{i},{bits},{energy:.12g}")

    if solved_by_id:
        aggregated_root = _aggregate_tree_solutions(root_id, nodes, solved_by_id, full_qubo.mat, row_map, col_map)
        if aggregated_root is not None and aggregated_root.solutions is not None and not aggregated_root.solutions.empty:
            full_qubo.solutions = aggregated_root.solutions
            agg_cols = [int(v) for v in full_qubo.cols_idx]
            for i, row in aggregated_root.solutions.reset_index(drop=True).iterrows():
                bits = _bitstring_from_row(row, agg_cols)
                try:
                    energy = float(row["energy"])
                except Exception:
                    energy = float("nan")
                lines.append(f"{root_id},aggregate,{i},{bits},{energy:.12g}")

    Path("aggregate.solutions.csv").write_text("\n".join(lines), encoding="utf-8")

    save_qubo(Path(args.output_qubo), full_qubo)


if __name__ == "__main__":
    main()
