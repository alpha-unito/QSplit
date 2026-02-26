import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

from qsplit.aggregation.aggregate_recursive import aggregate_solutions
from qsplit.cwl.cli.utils import (
    bitstring_from_row,
    build_index_maps,
    child_sort_key,
    load_qubo,
    load_solved_qubos,
    map_indices,
    parse_solved_paths,
)
from qsplit.qubo import QUBO


def _workspace_roots_from_paths(paths: List[Path]) -> List[Path]:
    marker = "/streamflow_workdir/"
    roots: set[Path] = set()
    for path in paths:
        text = str(path)
        if marker not in text:
            continue
        prefix = text.split(marker, 1)[0] + marker.rstrip("/")
        roots.add(Path(prefix))
    return sorted(roots)


def _discover_solved_for_instance(
    workspace_roots: List[Path],
    expected_instance: str,
    required_nodes: Set[str],
    known_paths: Set[Path],
    required_node_specs: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]],
) -> List[Tuple[Path, str, QUBO]]:
    if not expected_instance or not required_nodes:
        return []

    candidates: Dict[str, Tuple[float, Path, QUBO]] = {}

    for root in workspace_roots:
        if not root.exists():
            continue
        for pattern in ("solved.pkl", "root_*.pkl"):
            for candidate in root.rglob(pattern):
                try:
                    resolved = candidate.resolve()
                except Exception:
                    resolved = candidate
                if resolved in known_paths:
                    continue
                try:
                    qubo = load_qubo(candidate)
                except Exception:
                    continue
                if not isinstance(qubo, QUBO):
                    continue
                source_instance = str(getattr(qubo, "instance_id", "") or "").strip()
                if source_instance != expected_instance:
                    continue
                node_id = str(getattr(qubo, "node_id", "") or candidate.stem)
                if node_id not in required_nodes:
                    continue
                spec = required_node_specs.get(node_id)
                if spec is not None:
                    rows_expected, cols_expected = spec
                    rows_found = tuple(int(v) for v in getattr(qubo, "rows_idx", []))
                    cols_found = tuple(int(v) for v in getattr(qubo, "cols_idx", []))
                    if rows_found != rows_expected or cols_found != cols_expected:
                        continue
                df = getattr(qubo, "solutions", None)
                if df is None or getattr(df, "empty", True):
                    continue
                try:
                    mtime = candidate.stat().st_mtime
                except Exception:
                    mtime = 0.0
                current = candidates.get(node_id)
                if current is None or mtime > current[0]:
                    candidates[node_id] = (mtime, candidate, qubo)

    discovered: List[Tuple[Path, str, QUBO]] = []
    for node_id in sorted(candidates):
        _, path, qubo = candidates[node_id]
        discovered.append((path, node_id, qubo))
    return discovered


def build_node_qubo(node: Dict, full_mat: np.ndarray, row_map: Dict[int, int], col_map: Dict[int, int]) -> QUBO | None:
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
    order: List[str] = []
    stack: List[str] = [root_id]
    seen: set[str] = set()

    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        order.append(node_id)
        children = nodes.get(node_id, {}).get("children") or []
        for cid in children:
            stack.append(cid)

    results: Dict[str, QUBO | None] = {}

    for node_id in reversed(order):
        node = nodes.get(node_id, {})
        children = node.get("children") or []
        if not children:
            results[node_id] = solved_qubos.get(node_id)
            continue
        child_qubos = [results.get(cid) for cid in sorted(children, key=child_sort_key)]
        if any(q is None or q.solutions is None or getattr(q.solutions, "empty", True) for q in child_qubos):
            results[node_id] = None
            continue
        node_qubo = build_node_qubo(node, full_mat, row_map, col_map)
        results[node_id] = aggregate_solutions(child_qubos, node_qubo) if node_qubo else None

    return results.get(root_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate QSplit outputs")
    parser.add_argument("--input-qubo", required=True, help="initial_qubo.pkl (full original QUBO)")
    parser.add_argument("--tree-file", required=True, help="tree.json")
    parser.add_argument("--solved-list", action="extend", nargs="+", default=[], help="Solved sub-QUBO")
    args = parser.parse_args()

    tree = json.loads(Path(args.tree_file).read_text(encoding="utf-8"))
    nodes: Dict[str, Dict] = tree.get("nodes", {})
    root_id = tree.get("root", "root")

    full_qubo = load_qubo(Path(args.input_qubo))

    needed_idx = set()
    for node in nodes.values():
        needed_idx.update(int(idx) for idx in (node.get("rows_idx") or []))
        needed_idx.update(int(idx) for idx in (node.get("cols_idx") or []))

    rows_missing = needed_idx - set(full_qubo.rows_idx)
    cols_missing = needed_idx - set(full_qubo.cols_idx)
    missing_idx = sorted(rows_missing | cols_missing)
    for idx in missing_idx:
        full_qubo.mat = np.pad(full_qubo.mat, ((0, 1), (0, 1)), mode="constant")
        full_qubo.rows_idx = np.append(full_qubo.rows_idx, idx)
        full_qubo.cols_idx = np.append(full_qubo.cols_idx, idx)
        full_qubo.problem_size = full_qubo.mat.shape[0]
    row_map, col_map = build_index_maps(full_qubo.rows_idx, full_qubo.cols_idx)

    solved_paths = parse_solved_paths(args.solved_list)
    solved_entries_raw, _ = load_solved_qubos(solved_paths)

    expected_instance = str(getattr(full_qubo, "instance_id", "") or "").strip()
    solved_entries: List[Tuple[Path, str, QUBO]] = []
    skipped_instance: List[Tuple[Path, str, str]] = []
    for path, node_id, qubo in solved_entries_raw:
        source_instance = str(getattr(qubo, "instance_id", "") or "").strip()
        if expected_instance and source_instance and source_instance != expected_instance:
            skipped_instance.append((path, node_id, source_instance))
            continue
        solved_entries.append((path, node_id, qubo))

    source_instances = sorted(
        {
            str(getattr(qubo, "instance_id", "") or "").strip()
            for _, _, qubo in solved_entries_raw
            if str(getattr(qubo, "instance_id", "") or "").strip()
        }
    )
    if source_instances:
        print(
            "QSPLIT AGGREGATE solved_instances="
            + ",".join(source_instances)
            + f" expected_instance={expected_instance or 'unknown'}",
            flush=True,
        )
    if skipped_instance:
        print(
            f"QSPLIT AGGREGATE skipped_solved_due_to_instance_mismatch={len(skipped_instance)}",
            flush=True,
        )

    solved_by_id: Dict[str, QUBO] = {}
    duplicate_nodes: set[str] = set()
    for _, node_id, qubo in solved_entries:
        if node_id in solved_by_id:
            duplicate_nodes.add(node_id)
            continue
        solved_by_id[node_id] = qubo
    if duplicate_nodes:
        print(
            "QSPLIT AGGREGATE duplicate_node_ids_in_solved_list=" + ",".join(sorted(duplicate_nodes)),
            flush=True,
        )

    leaf_nodes = sorted(node_id for node_id, node in nodes.items() if not (node.get("children") or []))
    leaf_specs: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {
        node_id: (
            tuple(int(v) for v in (nodes.get(node_id, {}).get("rows_idx") or [])),
            tuple(int(v) for v in (nodes.get(node_id, {}).get("cols_idx") or [])),
        )
        for node_id in leaf_nodes
    }
    missing_leaf_nodes = sorted(node_id for node_id in leaf_nodes if node_id not in solved_by_id)
    if missing_leaf_nodes:
        workspace_roots = _workspace_roots_from_paths(
            [Path(args.input_qubo), Path(args.tree_file), *solved_paths]
        )
        discovered = _discover_solved_for_instance(
            workspace_roots=workspace_roots,
            expected_instance=expected_instance,
            required_nodes=set(missing_leaf_nodes),
            known_paths={path.resolve() for path, _, _ in solved_entries},
            required_node_specs=leaf_specs,
        )
        if discovered:
            for path, node_id, qubo in discovered:
                if node_id in solved_by_id:
                    continue
                solved_entries.append((path, node_id, qubo))
                solved_by_id[node_id] = qubo
            print(
                f"QSPLIT AGGREGATE recovered_solved_from_workspace={len(discovered)} "
                f"workspace_roots={','.join(str(root) for root in workspace_roots)}",
                flush=True,
            )
            missing_leaf_nodes = sorted(node_id for node_id in leaf_nodes if node_id not in solved_by_id)
        if missing_leaf_nodes:
            preview = ",".join(missing_leaf_nodes[:10])
            message = (
                f"QSPLIT AGGREGATE missing_solved_leaf_nodes={len(missing_leaf_nodes)} "
                f"preview={preview}"
            )
            print(message, flush=True)
            if expected_instance:
                raise RuntimeError(message)

    if expected_instance and not solved_by_id:
        raise RuntimeError(
            f"QSPLIT AGGREGATE no solved QUBOs available for expected_instance={expected_instance}"
        )

    rows: List[Tuple[str, str, str, str]] = []
    best_by_node_backend: Dict[Tuple[str, str], List[Tuple[float, str]]] = {}

    for _, node_id, qubo in solved_entries:
        backend = str(getattr(qubo, "backend", "") or "")
        if not backend:
            backend = str(nodes.get(node_id, {}).get("backend", ""))

        df = getattr(qubo, "solutions", None)
        if df is None or getattr(df, "empty", True):
            continue

        bit_cols = [int(v) for v in qubo.cols_idx]

        for _, row in df.reset_index(drop=True).iterrows():
            bits = bitstring_from_row(row, bit_cols)
            try:
                energy = float(row["energy"])
            except Exception:
                energy = float("nan")
            if np.isnan(energy) and backend != "dummy":
                continue
            best_by_node_backend.setdefault((node_id, backend), []).append((energy, bits))

    for (node_id, backend), entries in best_by_node_backend.items():
        entries.sort(key=lambda e: e[0])
        for energy, bits in entries[:3]:
            rows.append((node_id, backend, bits, f"{energy:.12g}"))

    if solved_by_id:
        aggregated_root = aggregate_tree_solutions(root_id, nodes, solved_by_id, full_qubo.mat, row_map, col_map)
        if (
            aggregated_root is not None
            and aggregated_root.solutions is not None
            and not aggregated_root.solutions.empty
        ):
            full_qubo.solutions = aggregated_root.solutions
            agg_cols = [int(v) for v in full_qubo.cols_idx]
            agg_entries: List[Tuple[float, str]] = []
            for _, row in aggregated_root.solutions.reset_index(drop=True).iterrows():
                bits = bitstring_from_row(row, agg_cols)
                try:
                    energy = float(row["energy"])
                except Exception:
                    energy = float("nan")
                if np.isnan(energy):
                    continue
                agg_entries.append((energy, bits))
            agg_entries.sort(key=lambda e: e[0])
            for energy, bits in agg_entries[:3]:
                rows.append((root_id, "aggregate", bits, f"{energy:.12g}"))
        else:
            message = (
                f"QSPLIT AGGREGATE root aggregation failed for instance="
                f"{expected_instance or 'unknown'} root={root_id}"
            )
            print(message, flush=True)
            if expected_instance:
                raise RuntimeError(message)

    rows.sort(key=lambda r: r[0])
    lines = ["node_id,backend,bitstring,energy"]
    lines.extend(f"{node_id},{backend},{bitstring},{energy}" for node_id, backend, bitstring, energy in rows)
    Path("solutions.csv").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
