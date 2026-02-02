import argparse
import os
from typing import Callable
from qsplit.qubo import QUBO
from qsplit.cwl.cli.utils import load_qubo, save_qubo


def resolve_backend(raw: str) -> str:
    b = (raw or "").strip().lower()
    if not b or b in {"auto", "any"}:
        env_backend = os.getenv("QSPLIT_BACKEND", "").strip().lower()
        if env_backend:
            return env_backend
        return "classic"
    if b in {"dummy_solver", "dummy-solver"}:
        return "dummy"
    return b


def load_solver(backend: str) -> Callable:
    b = (backend or "").strip().lower()

    if b == "dummy":
        from qsplit.adapters.dummy import solve as dummy_solve
        return dummy_solve

    if b == "classic":
        from qsplit.adapters.dwave.dwave_sa import solve as solver_fn
        return solver_fn

    if b == "dwave":
        from qsplit.adapters.dwave.dwave_sa import solve as solver_fn
        return solver_fn

    if b == "ibm":
        from qsplit.adapters.ibm.ibm_qaoa_cpu_noiseless import solve as solver_fn
        # from qsplit.adapters.dwave.dwave_sa import solve as solver_fn
        return solver_fn
    
    if b == "ibm_gpu":
        from qsplit.adapters.ibm.ibm_qaoa_gpu_noiseless import solve as solver_fn
        # from qsplit.adapters.dwave.dwave_sa import solve as solver_fn
        return solver_fn

    if b == "iqm":
        from qsplit.adapters.dwave.dwave_sa import solve as solver_fn
        return solver_fn

    from qsplit.adapters.all_zero import solve as solver_fn
    return solver_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve a sub-QUBO on a specified backend")
    parser.add_argument("--input-qubo", required=True)
    parser.add_argument("--output-qubo", required=True)
    parser.add_argument("--backend", default="auto")
    args = parser.parse_args()

    qubo = load_qubo(args.input_qubo)
    if not isinstance(qubo, QUBO):
        raise TypeError(f"Expected QUBO object in {args.input_qubo}, got {type(qubo)}")

    backend = resolve_backend(args.backend)
    solver_fn = load_solver(backend)
    df = solver_fn(qubo)
    df = df.nsmallest(1, "energy")
    qubo.solutions = df
    qubo.backend = backend
    save_qubo(args.output_qubo, qubo)
    if not os.path.exists(args.output_qubo):
        raise SystemExit(f"Output not created: {args.output_qubo}")


if __name__ == "__main__":
    main()
