from qsplit.qubo import QUBO
from qsplit.refinement.propagation import collect_beliefs, condition_subproblem


def refine_problems(subproblems: list[QUBO], qubo: QUBO) -> list[QUBO]:
    beliefs = collect_beliefs(subproblems, qubo)
    windows = dict.fromkeys(
        tuple(idx for idx in indices if idx >= 0) for sub in subproblems for indices in (sub.rows_idx, sub.cols_idx)
    )
    return [condition_subproblem(qubo, list(window), beliefs) for window in windows if window]
