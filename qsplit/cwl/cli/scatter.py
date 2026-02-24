import argparse
import importlib
import os
from typing import Callable

import pandas as pd

from qsplit.cwl.cli.utils import load_qubo, save_qubo
from qsplit.qubo import QUBO


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve a sub-QUBO on the selected backend")
    parser.add_argument("--input-qubo", required=True)
    parser.add_argument("--output-qubo", required=True)
    args = parser.parse_args()

    backend = resolve_backend(os.getenv("QSPLIT_BACKEND", ""))

    qubo = load_qubo(args.input_qubo)
    if not isinstance(qubo, QUBO):
        raise TypeError(f"Expected QUBO object in {args.input_qubo}, got {type(qubo)}")

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
    if not os.path.exists(args.output_qubo):
        raise SystemExit(f"Output not created: {args.output_qubo}")


if __name__ == "__main__":
    main()
