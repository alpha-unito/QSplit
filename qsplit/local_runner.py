import os
import warnings
from collections.abc import Callable
from copy import deepcopy

import numpy as np

from qsplit.adapters.all_zero import solve as zero_solve
from qsplit.adapters.dummy import solve as dummy_solve
from qsplit.adapters.dwave.dwave_sa import solve

# from qsplit.adapters.ibm.ibm_default import solve
from qsplit.aggregation.aggregate_k_interactions import aggregate_solutions as aggregate_solutions_interactions
from qsplit.aggregation.aggregate_linear import aggregate_solutions as aggregate_solutions_linear
from qsplit.aggregation.aggregate_linear_belief_propagation import aggregate_solutions as aggregate_solutions_linear_bp
from qsplit.aggregation.aggregate_quadtree import aggregate_solutions as aggregate_solutions_quadtree
from qsplit.aggregation.aggregate_recursive import aggregate_solutions as aggregate_solutions_recursive
from qsplit.aggregation.aggregate_recursive import aggregate_solutions_trivial
from qsplit.aggregation.aggregate_recursive_graph import aggregate_solutions as aggregate_solutions_recursive_graph
from qsplit.halting_heuristic.stop import is_empty, is_sparse
from qsplit.qubo import QUBO
from qsplit.refinement.refine_conditioned import refine_solutions as refine_solutions_conditioned
from qsplit.refinement.refine_linear import refine_problems as refine_problems_linear
from qsplit.refinement.refine_quadtree import refine_problems as refine_problems_quadtree
from qsplit.splitting.split_k_interactions import split_problem as split_problem_interactions
from qsplit.splitting.split_linear import split_problem as split_problem_linear
from qsplit.splitting.split_quadtree import split_problem as split_problem_quadtree
from qsplit.splitting.split_recursive import split_problem as split_problem_recursive
from qsplit.splitting.split_recursive_graph import split_problem as split_problem_recursive_graph

warnings.warn(
    "local_runner module is deprecated. This was the legacy method for using QSplit. "
    "It is now recommended to use StreamFlow to leverage multiple quantum backends. "
    "For more information on using StreamFlow with QSplit, please refer to the README.md file.",
    DeprecationWarning,
    stacklevel=1,
)

LOGICAL_EXPANSION = False
BP = False


def logical_expansion(subs: tuple[QUBO, QUBO, QUBO]) -> tuple[QUBO, QUBO, QUBO]:
    ul, ur, lr = subs
    hints = __extract_logical_hints(ur)
    if not hints:
        return subs

    ul_diagonal = {idx: pos for pos, idx in enumerate(ul.rows_idx) if idx >= 0}
    lr_diagonal = {idx: pos for pos, idx in enumerate(lr.rows_idx) if idx >= 0}

    for row_pos, row_idx in enumerate(ur.rows_idx):
        for col_pos, col_idx in enumerate(ur.cols_idx):
            coefficient = ur.mat[row_pos, col_pos]
            if coefficient == 0:
                continue

            col_hint = hints.get(col_idx)
            if row_idx in ul_diagonal and col_hint is not None:
                ul_pos = ul_diagonal[row_idx]
                ul.mat[ul_pos, ul_pos] += coefficient * col_hint

            row_hint = hints.get(row_idx)
            if col_idx in lr_diagonal and row_hint is not None:
                lr_pos = lr_diagonal[col_idx]
                lr.mat[lr_pos, lr_pos] += coefficient * row_hint

    return subs


def __extract_logical_hints(qubo: QUBO) -> dict[int, float]:
    if qubo.solutions is None or qubo.solutions.empty or "energy" not in qubo.solutions.columns:
        return {}

    best_energy = qubo.solutions["energy"].min()
    best_solutions = qubo.solutions[qubo.solutions["energy"] == best_energy]
    hints = {}

    for col in best_solutions.columns:
        if col == "energy" or col < 0:
            continue

        values = best_solutions[col].replace([np.inf, -np.inf], np.nan).dropna()
        if not values.empty:
            hints[col] = float(values.mean())

    return hints


def qsplit_sampler_recursive(qubo: QUBO) -> QUBO:
    if is_empty(qubo):
        qubo.solutions = dummy_solve(qubo)
        return qubo
    if (qubo.problem_size <= int(os.environ["CUT_DIM"])) or is_sparse(qubo):
        qubo.solutions = solve(qubo)
        return qubo

    subs = split_problem_recursive(qubo)
    if LOGICAL_EXPANSION:
        subs[1].solutions = qsplit_sampler_recursive(subs[1]).solutions
        subs = logical_expansion(subs)
        subs[0].solutions = qsplit_sampler_recursive(subs[0]).solutions
        subs[2].solutions = qsplit_sampler_recursive(subs[2]).solutions
        return aggregate_solutions_trivial(subs[0], subs[2], qubo)
    else:
        subs[0].solutions = qsplit_sampler_recursive(subs[0]).solutions
        subs[1].solutions = qsplit_sampler_recursive(subs[1]).solutions
        subs[2].solutions = qsplit_sampler_recursive(subs[2]).solutions
        return aggregate_solutions_recursive(subs, qubo)


def qsplit_sampler_iterative(qubo: QUBO) -> QUBO:
    subs = split_problem_linear(qubo)
    for p in subs:
        if is_empty(p):
            p.solutions = dummy_solve(p)
        else:
            p.solutions = solve(p)
    return aggregate_solutions_linear_bp(subs, qubo) if BP else aggregate_solutions_linear(subs, qubo)


def qsplit_sampler_interactions(qubo: QUBO) -> QUBO:
    subs = split_problem_interactions(qubo)
    for p in subs:
        if is_empty(p):
            p.solutions = dummy_solve(p)
        else:
            p.solutions = solve(p)
    return aggregate_solutions_interactions(subs, qubo)


def qsplit_sampler_graph_partitioning(qubo: QUBO) -> QUBO:
    subs = split_problem_recursive_graph(qubo)
    for p in subs:
        if is_empty(p):
            p.solutions = dummy_solve(p)
        else:
            p.solutions = solve(p)
    return aggregate_solutions_recursive_graph(subs, qubo)


def qsplit_sampler_quadtree(qubo: QUBO) -> QUBO:
    subs = split_problem_quadtree(qubo)
    for p in subs:
        if is_empty(p):
            p.solutions = dummy_solve(p)
        else:
            p.solutions = solve(p)
    return aggregate_solutions_quadtree(subs, qubo)


def _qsplit_sampler_refined(
    qubo: QUBO,
    split: Callable[[QUBO], list[QUBO]],
    aggregate: Callable[[list[QUBO], QUBO], QUBO],
    refinement: Callable[[list[QUBO], QUBO], list[QUBO]] | None,
    loops: int,
) -> QUBO:
    tolerance = float(os.environ.get("REFINEMENT_TOLERANCE", "1e-9"))
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("REFINEMENT_TOLERANCE must be finite and non-negative")
    patience = int(os.environ.get("REFINEMENT_PATIENCE", "10" if refinement is None else "1"))
    if patience <= 0:
        raise ValueError("REFINEMENT_PATIENCE must be positive")
    rng = np.random.default_rng(int(os.environ.get("REFINEMENT_SEED", "0"))) if refinement is None else None
    block_size = int(os.environ["CUT_DIM"])
    if split is split_problem_quadtree:
        block_size -= block_size % 2
    if block_size <= 0:
        raise ValueError("Refinement requires a positive effective CUT_DIM")

    original = deepcopy(qubo)
    subs = split(qubo)
    best_solutions = None
    best_energy = np.inf
    history = []
    working_solutions = None
    stalled = 0

    for iteration in range(loops + 1):
        if iteration and refinement is None:
            original.solutions = working_solutions.copy(deep=True)
            candidates = refine_solutions_conditioned(original, solve, block_size, rng)
        else:
            if iteration:
                original.solutions = best_solutions.copy(deep=True)
                subs = refinement(subs, original)
            for sub in subs:
                empty = not np.any(sub.mat) if getattr(sub, "macro_members", {}) else is_empty(sub)
                if empty:
                    sub.solutions = zero_solve(sub)
                    sub.solutions["energy"] = sub.offset
                else:
                    sub.solutions = solve(sub)
            result = aggregate(subs, qubo)
            candidates = result.solutions.copy(deep=True)
        if candidates.empty:
            raise ValueError("Refinement requires a complete global solution")
        for label, candidate in candidates.iterrows():
            rows = np.array([candidate[idx] if idx >= 0 else 0 for idx in original.rows_idx])
            cols = np.array([candidate[idx] if idx >= 0 else 0 for idx in original.cols_idx])
            if not np.all(np.isin(rows, [0, 1])) or not np.all(np.isin(cols, [0, 1])):
                raise ValueError("Refinement requires binary assignments for every real variable")
            candidates.loc[label, "energy"] = float(rows @ original.mat @ cols + original.offset)
        energy = float(candidates["energy"].min())
        if not np.isfinite(energy):
            raise ValueError("Refinement requires finite global energies")
        history.append(energy)
        improvement = best_energy - energy
        if best_solutions is None or energy < best_energy:
            best_solutions = candidates
            best_energy = energy
        working_solutions = candidates
        stalled = stalled + 1 if improvement <= tolerance else 0
        if not subs or stalled >= patience:
            break

    qubo.solutions = best_solutions
    qubo.refinement_history = history
    return qubo


def _select_refinement(
    refinement: Callable[[list[QUBO], QUBO], list[QUBO]] | None,
    consensus_refinement: Callable[[list[QUBO], QUBO], list[QUBO]],
) -> Callable[[list[QUBO], QUBO], list[QUBO]] | None:
    if refinement is not None:
        return refinement
    method = os.environ.get("REFINEMENT_METHOD", "conditioned")
    if method == "consensus":
        return consensus_refinement
    if method != "conditioned":
        raise ValueError("REFINEMENT_METHOD must be 'conditioned' or 'consensus'")
    return None


def qsplit_sampler_refined_iterative(
    qubo: QUBO,
    *,
    refinement: Callable[[list[QUBO], QUBO], list[QUBO]] | None = None,
) -> QUBO:
    loops = int(os.environ.get("REFINEMENT_LOOPS", "0"))
    if loops <= 0:
        return qsplit_sampler_iterative(qubo)
    refinement = _select_refinement(refinement, refine_problems_linear)
    aggregate = aggregate_solutions_linear_bp if BP else aggregate_solutions_linear
    return _qsplit_sampler_refined(qubo, split_problem_linear, aggregate, refinement, loops)


def qsplit_sampler_refined_quadtree(
    qubo: QUBO,
    *,
    refinement: Callable[[list[QUBO], QUBO], list[QUBO]] | None = None,
) -> QUBO:
    loops = int(os.environ.get("REFINEMENT_LOOPS", "0"))
    if loops <= 0:
        return qsplit_sampler_quadtree(qubo)
    refinement = _select_refinement(refinement, refine_problems_quadtree)
    return _qsplit_sampler_refined(qubo, split_problem_quadtree, aggregate_solutions_quadtree, refinement, loops)
