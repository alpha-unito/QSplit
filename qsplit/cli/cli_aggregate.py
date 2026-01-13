import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd
from qsplit.aggregation.aggregate_recursive import aggregate_solutions
from .io_utils import load_qubo
from qsplit.qubo import QUBO


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


def _build_index_maps(rows_idx: np.ndarray, cols_idx: np.ndarray) -> tuple[dict[int, int], dict[int, int]]:
    row_map = {int(v): i for i, v in enumerate(rows_idx)}
    col_map = {int(v): i for i, v in enumerate(cols_idx)}
    return row_map, col_map


def _map_indices(idxs: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    return np.array([mapping[int(v)] for v in idxs], dtype=int)


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
    parser = argparse.ArgumentParser(description="Aggregate QSplit outputs")
    parser.add_argument("--input-qubo", required=True, help="initial_qubo.pkl (full original QUBO)")
    parser.add_argument("--tree-file", required=True, help="tree.json")
    parser.add_argument("--solved-list", default="", help="Comma-separated list of solved sub-QUBO pkls")
    args = parser.parse_args()

    tree = json.loads(Path(args.tree_file).read_text(encoding="utf-8"))
    nodes: Dict[str, Dict] = tree.get("nodes", {})
    root_id = tree.get("root", "root")

    full_qubo = load_qubo(Path(args.input_qubo))
    if not isinstance(full_qubo, QUBO):
        raise TypeError(f"Expected QUBO in {args.input_qubo}, got {type(full_qubo)}")
    row_map, col_map = _build_index_maps(full_qubo.rows_idx, full_qubo.cols_idx)

    solved_paths = _parse_solved_paths(args.solved_list)
    solved_entries, solved_by_id = _load_solved_qubos(solved_paths)

    rows: List[Tuple[str, str, str, str]] = []

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
            rows.append((node_id, backend, bits, f"{energy:.12g}"))

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
                rows.append((root_id, "aggregate", f"{i},{bits}", f"{energy:.12g}"))

    rows.sort(key=lambda r: r[0])
    lines = ["node_id,backend,bitstring,energy"]
    lines.extend(f"{node_id},{backend},{bitstring},{energy}" for node_id, backend, bitstring, energy in rows)
    Path("aggregate.solutions.csv").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
