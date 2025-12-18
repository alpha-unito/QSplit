import argparse
import os
import sys
from typing import Callable

from .io_utils import load_qubo, save_qubo
from .qubo import QUBO


def _warn(msg: str) -> None:
    print(f"[qsplit.cli_dwave_solve] {msg}", file=sys.stderr)


def _load_solver(backend: str) -> Callable:
    b = (backend or "").strip().lower()

    if os.getenv("SOLVER_TAG", "").strip().lower() == "classic":
        b = "classic"

    if b == "classic":
        from .adapters.all_zero import solve as solver_fn
        return solver_fn

    if b == "dwave":
        try:
            from .adapters.dwave.dwave_sa import solve as solver_fn
            return solver_fn
        except Exception as e1:
            _warn(f"D-Wave QA solver not available ({type(e1).__name__}: {e1}). Trying SA...")
            try:
                from .adapters.dwave.dwave_sa import solve as solver_fn
                return solver_fn
            except Exception as e2:
                _warn(f"D-Wave SA solver not available ({type(e2).__name__}: {e2}). Falling back to classic.")
                from .adapters.all_zero import solve as solver_fn
                return solver_fn

    if b == "ibm":
        try:
            from .adapters.ibm.ibm_qaoa_cpu_noiseless import solve as solver_fn
            return solver_fn
        except ModuleNotFoundError as e:
            _warn(f"IBM solver dependencies missing ({e}). Falling back to classic.")
            from .adapters.all_zero import solve as solver_fn
            return solver_fn
        except Exception as e:
            _warn(f"IBM solver not available ({type(e).__name__}: {e}). Falling back to classic.")
            from .adapters.all_zero import solve as solver_fn
            return solver_fn

    if b == "iqm":
        try:
            from .adapters.dwave.dwave_sa import solve as solver_fn
            return solver_fn
        except Exception as e:
            _warn(f"IQM fallback SA not available ({type(e).__name__}: {e}). Falling back to classic.")
            from .adapters.all_zero import solve as solver_fn
            return solver_fn

    _warn(f"Unknown backend '{backend}', falling back to classic.")
    from .adapters.all_zero import solve as solver_fn
    return solver_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve a sub-QUBO on a specified backend")
    parser.add_argument("--input-qubo", required=True)
    parser.add_argument("--output-qubo", required=True)
    parser.add_argument("--backend", required=True)
    args = parser.parse_args()

    qubo = load_qubo(args.input_qubo)
    if not isinstance(qubo, QUBO):
        raise TypeError(f"Expected QUBO object in {args.input_qubo}, got {type(qubo)}")

    solver_fn = _load_solver(args.backend)
    df = solver_fn(qubo)
    qubo.solutions = df
    save_qubo(args.output_qubo, qubo)


if __name__ == "__main__":
    main()