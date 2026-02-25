import argparse
import importlib
import os
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from qsplit.cwl.cli.utils import load_qubo, save_qubo
from qsplit.qubo import QUBO

IQM_CACHE_MISS_EXIT_CODE = 75


def resolve_backend(raw: str) -> str:
    b = (raw or "").strip().lower()
    if not b or b in {"auto", "any"}:
        env_backend = os.getenv("QSPLIT_BACKEND", "").strip().lower()
        if env_backend:
            return env_backend
        return "dwave"
    if b in {"dummy_solver", "dummy-solver"}:
        return "dummy"
    return b


def load_solver(backend: str) -> Callable:
    b = (backend or "").strip().lower()

    module_override = os.getenv("QSPLIT_SOLVER_MODULE", "").strip()
    if module_override:
        mod = importlib.import_module(module_override)
        solver_fn = getattr(mod, "solve", None)
        if callable(solver_fn):
            return solver_fn
        raise AttributeError(f"{module_override}.solve is not callable")

    if b == "dummy":
        from qsplit.adapters.dummy import solve as dummy_solve

        return dummy_solve

    if b == "dwave":
        from qsplit.adapters.dwave.dwave_sa import solve as solver_fn

        return solver_fn

    if b == "ibm":
        from qsplit.adapters.ibm.ibm_qaoa_cpu_noiseless import solve as solver_fn

        return solver_fn

    if b == "cudaq":
        from qsplit.adapters.nvidia.cudaq_qaoa import solve as solver_fn

        return solver_fn

    if b == "ibm_gpu":
        from qsplit.adapters.ibm.ibm_qaoa_gpu_noiseless import solve as solver_fn

        return solver_fn

    if b == "iqm":
        from qsplit.adapters.iqm.iqm_qaoa_q import solve as solver_fn

        return solver_fn

    if b == "quantinuum_h2":
        from qsplit.adapters.quantinuum.quantinuum_h2 import solve as solver_fn

        return solver_fn

    if b == "quantinuum_h2e":
        from qsplit.adapters.quantinuum.quantinuum_h2e import solve as solver_fn

        return solver_fn

    from qsplit.adapters.all_zero import solve as solver_fn

    return solver_fn


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _looks_like_project_root(path: Path) -> bool:
    return (path / "qsplit").is_dir() and (path / "streamflow").is_dir()


def _candidate_launch_dirs() -> list[Path]:
    candidates: list[Path] = []
    for env_name in ("QSPLIT_LAUNCH_DIR", "QSPLIT_PROJECT_ROOT", "PWD", "OLDPWD", "INIT_CWD"):
        raw = os.getenv(env_name, "").strip()
        if raw:
            candidates.append(Path(raw).expanduser())
    return candidates


def _resolve_iqm_subproblem_dir() -> Path:
    base = Path("iqm_solutions")

    for candidate in _candidate_launch_dirs():
        resolved = candidate.resolve()
        if _looks_like_project_root(resolved):
            return resolved / base

    repo_root = _repo_root().resolve()
    if _looks_like_project_root(repo_root):
        return repo_root / base

    for candidate in _candidate_launch_dirs():
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved / base

    return Path.cwd().resolve() / base


def _iqm_cache_enabled() -> bool:
    raw = (os.getenv("QSPLIT_IQM_SUBPROBLEM_CACHE", "1") or "").strip().lower()
    return raw not in {"0", "false", "no"}


def _iqm_cache_probe_only() -> bool:
    raw = (os.getenv("QSPLIT_IQM_CACHE_ONLY", "") or "").strip().lower()
    return raw in {"1", "true", "yes"}


def _safe_cache_component(value: str, fallback: str) -> str:
    text = (value or "").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or fallback


def _iqm_cache_coordinates(qubo: QUBO, input_qubo_path: str) -> tuple[str, str, str, str]:
    instance_id = _safe_cache_component(str(getattr(qubo, "instance_id", "")), "instance_unknown")
    node_id = _safe_cache_component(str(getattr(qubo, "node_id", "")), "node_unknown")
    if node_id == "node_unknown":
        node_id = _safe_cache_component(Path(input_qubo_path).stem, "node_unknown")
    quantum_computer = _safe_cache_component(os.getenv("IQM_QUANTUM_COMPUTER", ""), "qc_default")
    quantum_tune = _safe_cache_component(os.getenv("QUANTUM_TUNE_QAOA", ""), "qt_default")
    return instance_id, node_id, quantum_computer, quantum_tune


def _same_subproblem(a: QUBO, b: QUBO) -> bool:
    if a.mat.shape != b.mat.shape:
        return False
    if not np.array_equal(np.asarray(a.rows_idx), np.asarray(b.rows_idx)):
        return False
    if not np.array_equal(np.asarray(a.cols_idx), np.asarray(b.cols_idx)):
        return False
    if not np.allclose(np.asarray(a.mat), np.asarray(b.mat), atol=1e-12, rtol=0.0):
        return False
    return abs(float(getattr(a, "offset", 0.0)) - float(getattr(b, "offset", 0.0))) <= 1e-12


def _load_cached_iqm_dataframe(path: Path, expected_qubo: QUBO) -> pd.DataFrame | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        cached = load_qubo(path)
    except Exception:
        return None
    if not isinstance(cached, QUBO):
        return None
    if not _same_subproblem(cached, expected_qubo):
        return None
    df = getattr(cached, "solutions", None)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if "energy" not in df.columns:
        return None
    if bool(df["energy"].isna().all()):
        return None
    return df.copy(deep=True)


def _store_cached_iqm_solution(path: Path, qubo: QUBO) -> None:
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    save_qubo(tmp_path, qubo)
    os.replace(tmp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve a sub-QUBO on the selected backend")
    parser.add_argument("--input-qubo", required=True)
    parser.add_argument("--output-qubo", required=True)
    args = parser.parse_args()

    backend = resolve_backend(os.getenv("QSPLIT_BACKEND", ""))

    qubo = load_qubo(args.input_qubo)
    if not isinstance(qubo, QUBO):
        raise TypeError(f"Expected QUBO object in {args.input_qubo}, got {type(qubo)}")

    iqm_cache_path: Path | None = None
    iqm_cache_label: str | None = None
    if backend == "iqm" and _iqm_cache_enabled():
        cache_dir = _resolve_iqm_subproblem_dir()
        instance_id, node_id, qc_name, qt_mode = _iqm_cache_coordinates(qubo, args.input_qubo)
        instance_dir = cache_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        iqm_cache_path = instance_dir / f"{node_id}__qc_{qc_name}__qt_{qt_mode}.pkl"
        iqm_cache_label = f"instance={instance_id} node={node_id} qc={qc_name} qt={qt_mode}"
        cached_df = _load_cached_iqm_dataframe(iqm_cache_path, qubo)
        if cached_df is not None:
            qubo.solutions = cached_df
            qubo.backend = "iqm"
            save_qubo(args.output_qubo, qubo)
            print(f"QSPLIT IQM SUBPROBLEM CACHE HIT {iqm_cache_label} path={iqm_cache_path}", flush=True)
            if not os.path.exists(args.output_qubo):
                raise SystemExit(f"Output not created: {args.output_qubo}")
            return
        if _iqm_cache_probe_only():
            print(f"QSPLIT IQM SUBPROBLEM CACHE MISS {iqm_cache_label} path={iqm_cache_path}", flush=True)
            raise SystemExit(IQM_CACHE_MISS_EXIT_CODE)

    solver_fn = load_solver(backend)
    df = solver_fn(qubo)
    if df is None:
        raise RuntimeError(f"Solver returned no result for backend {backend}")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Solver for backend {backend} must return pandas.DataFrame, got {type(df)}")
    if df.empty:
        raise RuntimeError(f"Solver returned empty DataFrame for backend {backend}")
    if backend != "dummy":
        if "energy" not in df.columns:
            raise RuntimeError(f"Missing 'energy' column in solver output for backend {backend}")
        if bool(df["energy"].isna().all()):
            raise RuntimeError(f"All energies are NaN for backend {backend}")
    qubo.solutions = df
    qubo.backend = backend
    save_qubo(args.output_qubo, qubo)
    if backend == "iqm" and iqm_cache_path is not None:
        try:
            _store_cached_iqm_solution(iqm_cache_path, qubo)
            print(f"QSPLIT IQM SUBPROBLEM CACHE STORE {iqm_cache_label} path={iqm_cache_path}", flush=True)
        except Exception as exc:
            print(
                f"QSPLIT IQM SUBPROBLEM CACHE STORE FAILED {iqm_cache_label} path={iqm_cache_path} error={exc}",
                flush=True,
            )
    if not os.path.exists(args.output_qubo):
        raise SystemExit(f"Output not created: {args.output_qubo}")


if __name__ == "__main__":
    main()
