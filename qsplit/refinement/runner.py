from collections.abc import Callable
from copy import deepcopy

import numpy as np

from qsplit import _core, configuration
from qsplit.adapters.all_zero import solve as zero_solve
from qsplit.aggregation.aggregate_linear import aggregate_solutions
from qsplit.halting_heuristic.stop import is_empty
from qsplit.qubo import QUBO
from qsplit.refinement.refine_conditioned import refine_solutions
from qsplit.refinement.refine_mean_field import refine_problems as mean_field
from qsplit.refinement.refine_soft_consensus import refine_problems as soft_consensus


def solve_problems(subproblems, solve):
    for sub in subproblems:
        empty = not np.any(sub.mat) if getattr(sub, "macro_members", {}) else is_empty(sub)
        if empty:
            sub.solutions = zero_solve(sub)
            sub.solutions["energy"] = sub.offset
        else:
            sub.solutions = solve(sub)


def select_refinement(refinement, consensus_refinement):
    if refinement is not None:
        return refinement
    method = configuration.get("REFINEMENT_METHOD", "conditioned")
    if method == "conditioned":
        return None
    if method == "consensus":
        return consensus_refinement
    if method == "mean_field":
        return mean_field
    if method == "soft_consensus":
        return soft_consensus
    raise ValueError("REFINEMENT_METHOD must be 'conditioned', 'consensus', 'mean_field' or 'soft_consensus'")


@configuration.configured
def refine_result(
    qubo: QUBO,
    *,
    solve: Callable,
    subproblems: list[QUBO] | None = None,
    refinement: Callable | None = None,
    aggregate: Callable = aggregate_solutions,
    block_size: int | None = None,
) -> QUBO:
    loops = int(configuration.get("REFINEMENT_LOOPS", "0"))
    if loops <= 0:
        return qubo
    tolerance = float(configuration.get("REFINEMENT_TOLERANCE", "1e-9"))
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("REFINEMENT_TOLERANCE must be finite and non-negative")
    block_size = int(configuration.require("CUT_DIM")) if block_size is None else block_size
    if block_size <= 0:
        raise ValueError("Refinement requires a positive effective CUT_DIM")
    if qubo.solutions is None:
        raise ValueError("Refinement requires a complete global solution")
    original = deepcopy(qubo)
    subs = list(subproblems) if subproblems is not None else []
    if subproblems is None:
        ids = [int(idx) for idx in original.rows_idx if idx >= 0]
        for start in range(0, len(ids), block_size):
            window = ids[start : start + block_size]
            subs.append(QUBO(np.zeros((len(window), len(window))), np.array(window), np.array(window)))
    consensus = soft_consensus if any(getattr(sub, "macro_members", {}) for sub in subs) else mean_field
    refinement = select_refinement(refinement, consensus)
    patience = int(configuration.get("REFINEMENT_PATIENCE", "10" if refinement is None else "1"))
    if patience <= 0:
        raise ValueError("REFINEMENT_PATIENCE must be positive")
    rng = np.random.default_rng(int(configuration.get("REFINEMENT_SEED", "0"))) if refinement is None else None
    candidates = _core.validate_solutions(original.solutions.copy(deep=True), original)
    best_solutions = None
    best_energy = np.inf
    history = []
    stalled = 0
    for iteration in range(loops + 1):
        if iteration:
            if refinement is None:
                original.solutions = candidates.copy(deep=True)
                candidates = refine_solutions(original, solve, block_size, rng)
            else:
                original.solutions = best_solutions.copy(deep=True)
                subs = refinement(subs, original)
                solve_problems(subs, solve)
                candidates = aggregate(subs, deepcopy(original)).solutions.copy(deep=True)
            candidates = _core.validate_solutions(candidates, original)
        energy = float(candidates["energy"].min())
        if not np.isfinite(energy):
            raise ValueError("Refinement requires finite global energies")
        history.append(energy)
        improvement = best_energy - energy
        if best_solutions is None or energy < best_energy:
            best_solutions = candidates.copy(deep=True)
            best_energy = energy
        stalled = stalled + 1 if improvement <= tolerance else 0
        if not subs or stalled >= patience:
            break
    qubo.solutions = best_solutions
    qubo.refinement_history = history
    return qubo
