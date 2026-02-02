import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from qsplit.qubo import QUBO


def save_qubo(path: str | Path, qubo: QUBO) -> None:
    path = Path(path)
    with path.open("wb") as f:
        pickle.dump(qubo, f)


def load_qubo(path: str | Path) -> QUBO:
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)


def build_qubo_from_matrix(matrix_path: str) -> QUBO:
    mat = np.loadtxt(matrix_path, delimiter=",")
    n = mat.shape[0]
    qubo = QUBO(mat=mat, rows_idx=np.arange(n), cols_idx=np.arange(n))
    save_qubo("initial_qubo.pkl", qubo)
    return qubo


def parse_backend_cut_dims(spec: str) -> Dict[str, int]:
    res: Dict[str, int] = {}
    if not spec:
        return res
    for part in spec.split(","):
        part = part.strip()
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        try:
            res[k] = int(v)
        except ValueError:
            pass
    return res


def bitstring_from_row(row: pd.Series, ordered_cols: List[int]) -> str:
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


def parse_solved_paths(raw: str | List[str]) -> List[Path]:
    if not raw:
        return []
    parts: List[str] = []
    if isinstance(raw, list):
        for entry in raw:
            if not entry:
                continue
            for piece in str(entry).split(","):
                piece = piece.strip()
                if piece:
                    parts.append(piece)
    else:
        for piece in str(raw).split(","):
            piece = piece.strip()
            if piece:
                parts.append(piece)
    return [Path(p) for p in parts]


def load_solved_qubos(paths: Iterable[Path]) -> Tuple[List[Tuple[Path, str, QUBO]], Dict[str, QUBO]]:
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


def build_index_maps(rows_idx: np.ndarray, cols_idx: np.ndarray) -> tuple[dict[int, int], dict[int, int]]:
    row_map = {int(v): i for i, v in enumerate(rows_idx)}
    col_map = {int(v): i for i, v in enumerate(cols_idx)}
    return row_map, col_map


def map_indices(idxs: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    return np.array([mapping[int(v)] for v in idxs], dtype=int)


def child_sort_key(child_id: str) -> tuple[int, str]:
    return int(child_id.rsplit("_", 1)[1]), child_id
