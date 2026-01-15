import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from qsplit.aggregation.aggregate_recursive import aggregate_solutions
from .utils import bitstring_from_row, build_index_maps, child_sort_key, load_qubo, load_solved_qubos, map_indices, parse_solved_paths
from qsplit.qubo import QUBO


def build_node_qubo(
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
    row_pos = map_indices(rows, row_map)
    col_pos = map_indices(cols, col_map)
    mat = full_mat[np.ix_(row_pos, col_pos)]
    offset = float(node.get("offset", 0.0))
    return QUBO(mat, rows_idx=rows, cols_idx=cols, offset=offset)

def aggregate_tree_solutions(
    root_id: str,
    nodes: Dict[str, Dict],
    solved_qubos: Dict[str, QUBO],
    full_mat: np.ndarray,
    row_map: Dict[int, int],
    col_map: Dict[int, int],
) -> QUBO | None:
    def visit(node_id: str) -> QUBO | None:
        node = nodes.get(node_id, {})
        children = node.get("children") or []
        if not children:
            return solved_qubos.get(node_id)
        if len(children) != 3:
            raise ValueError(f"Node '{node_id}' expected 3 children, got {len(children)}.")
        child_qubos = [visit(cid) for cid in sorted(children, key=child_sort_key)]
        if any(q is None for q in child_qubos):
            return None
        if any(q.solutions is None or getattr(q.solutions, "empty", True) for q in child_qubos):
            return None
        node_qubo = build_node_qubo(node, full_mat, row_map, col_map)
        if node_qubo is None:
            return None
        return aggregate_solutions(child_qubos, node_qubo)

    return visit(root_id)


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
    row_map, col_map = build_index_maps(full_qubo.rows_idx, full_qubo.cols_idx)

    solved_paths = parse_solved_paths(args.solved_list)
    solved_entries, solved_by_id = load_solved_qubos(solved_paths)

    rows: List[Tuple[str, str, str, str]] = []

    for _, node_id, qubo in solved_entries:
        backend = str(nodes.get(node_id, {}).get("backend", ""))

        df = getattr(qubo, "solutions", None)
        if df is None or getattr(df, "empty", True):
            continue

        bit_cols = [int(v) for v in qubo.cols_idx]

        for i, row in df.reset_index(drop=True).iterrows():
            bits = bitstring_from_row(row, bit_cols)
            try:
                energy = float(row["energy"])
            except Exception:
                energy = float("nan")
            rows.append((node_id, backend, bits, f"{energy:.12g}"))

    if solved_by_id:
        aggregated_root = aggregate_tree_solutions(root_id, nodes, solved_by_id, full_qubo.mat, row_map, col_map)
        if aggregated_root is not None and aggregated_root.solutions is not None and not aggregated_root.solutions.empty:
            full_qubo.solutions = aggregated_root.solutions
            agg_cols = [int(v) for v in full_qubo.cols_idx]
            for i, row in aggregated_root.solutions.reset_index(drop=True).iterrows():
                bits = bitstring_from_row(row, agg_cols)
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