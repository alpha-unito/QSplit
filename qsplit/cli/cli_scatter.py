import argparse
from typing import Callable
from .io_utils import load_qubo, save_qubo
from qsplit.qubo import QUBO


def load_solver(backend: str) -> Callable:
    b = (backend or "").strip().lower()

    if b == "dummy":
        from qsplit.adapters.dummy import solve as dummy_solve
        return dummy_solve

    if b == "classic":
        # from adapters.all_zero import solve as solver_fn
        # return solver_fn
        from qsplit.adapters.dwave.dwave_sa import solve as solver_fn
        return solver_fn

    if b == "dwave":
        from qsplit.adapters.dwave.dwave_sa import solve as solver_fn
        return solver_fn

    if b == "ibm":
        # from qsplit.adapters.ibm.ibm_qaoa_cpu_noiseless import solve as solver_fn
        # return solver_fn
        from qsplit.adapters.dwave.dwave_sa import solve as solver_fn
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
    parser.add_argument("--backend", required=True)
    args = parser.parse_args()

    qubo = load_qubo(args.input_qubo)
    if not isinstance(qubo, QUBO):
        raise TypeError(f"Expected QUBO object in {args.input_qubo}, got {type(qubo)}")

    solver_fn = load_solver(args.backend)
    df = solver_fn(qubo)
    qubo.solutions = df
    save_qubo(args.output_qubo, qubo)


if __name__ == "__main__":
    main()