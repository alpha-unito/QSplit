import warnings
from collections.abc import Callable
from copy import deepcopy

from qsplit import configuration
from qsplit._core import logical_expansion as logical_expansion
from qsplit.adapters.all_zero import solve as zero_solve
from qsplit.adapters.dummy import solve as dummy_solve

# from qsplit.adapters.ibm.ibm_default import solve
from qsplit.aggregation.aggregate_k_interactions import aggregate_solutions as aggregate_solutions_interactions
from qsplit.aggregation.aggregate_linear import aggregate_solutions as aggregate_solutions_linear
from qsplit.aggregation.aggregate_linear_belief_propagation import aggregate_solutions as aggregate_solutions_linear_bp
from qsplit.aggregation.aggregate_quadtree import aggregate_solutions as aggregate_solutions_quadtree
from qsplit.aggregation.aggregate_recursive import aggregate_solutions as aggregate_solutions_recursive
from qsplit.aggregation.aggregate_recursive import aggregate_solutions_trivial
from qsplit.aggregation.aggregate_recursive_graph import aggregate_solutions as aggregate_solutions_recursive_graph
from qsplit.cwl.cli.scatter import load_solver
from qsplit.halting_heuristic.stop import is_empty, is_sparse
from qsplit.qubo import QUBO
from qsplit.refinement.refine_linear import refine_problems as refine_problems_linear
from qsplit.refinement.refine_mean_field import refine_problems as refine_problems_mean_field
from qsplit.refinement.refine_quadtree import refine_problems as refine_problems_quadtree
from qsplit.refinement.runner import refine_result, select_refinement, solve_problems
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


def solve(qubo):
    return load_solver(configuration.get("QSPLIT_BACKEND", "dwave"))(qubo)


LOGICAL_EXPANSION = False
BP = False


@configuration.configured
def qsplit_sampler_recursive(qubo: QUBO) -> QUBO:
    return _sample_recursive(qubo)


def _sample_recursive(qubo: QUBO, leaves: list[QUBO] | None = None) -> QUBO:
    if is_empty(qubo):
        qubo.solutions = dummy_solve(qubo) if leaves is None else zero_solve(qubo)
        if leaves is not None:
            qubo.solutions["energy"] = qubo.offset
            leaves.append(qubo)
        return qubo
    if (qubo.problem_size <= int(configuration.require("CUT_DIM"))) or is_sparse(qubo):
        qubo.solutions = solve(qubo)
        if leaves is not None:
            leaves.append(qubo)
        return qubo

    subs = split_problem_recursive(qubo)
    if LOGICAL_EXPANSION:
        subs[1].solutions = _sample_recursive(subs[1], leaves).solutions
        subs = logical_expansion(subs)
        subs[0].solutions = _sample_recursive(subs[0], leaves).solutions
        subs[2].solutions = _sample_recursive(subs[2], leaves).solutions
        return aggregate_solutions_trivial(subs[0], subs[2], qubo)
    else:
        subs[0].solutions = _sample_recursive(subs[0], leaves).solutions
        subs[1].solutions = _sample_recursive(subs[1], leaves).solutions
        subs[2].solutions = _sample_recursive(subs[2], leaves).solutions
        return aggregate_solutions_recursive(subs, qubo)


@configuration.configured
def qsplit_sampler_iterative(qubo: QUBO) -> QUBO:
    subs = split_problem_linear(qubo)
    for p in subs:
        if is_empty(p):
            p.solutions = dummy_solve(p)
        else:
            p.solutions = solve(p)
    return aggregate_solutions_linear_bp(subs, qubo) if BP else aggregate_solutions_linear(subs, qubo)


@configuration.configured
def qsplit_sampler_interactions(qubo: QUBO) -> QUBO:
    subs = split_problem_interactions(qubo)
    for p in subs:
        if is_empty(p):
            p.solutions = dummy_solve(p)
        else:
            p.solutions = solve(p)
    return aggregate_solutions_interactions(subs, qubo)


@configuration.configured
def qsplit_sampler_graph_partitioning(qubo: QUBO) -> QUBO:
    subs = split_problem_recursive_graph(qubo)
    for p in subs:
        if is_empty(p):
            p.solutions = dummy_solve(p)
        else:
            p.solutions = solve(p)
    return aggregate_solutions_recursive_graph(subs, qubo)


@configuration.configured
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
) -> QUBO:
    original = deepcopy(qubo)
    subs = split(qubo)
    solve_problems(subs, solve)
    original.solutions = aggregate(subs, qubo).solutions.copy(deep=True)
    block_size = int(configuration.require("CUT_DIM"))
    if split is split_problem_quadtree:
        block_size -= block_size % 2
    refinement_aggregate = aggregate
    if refinement is refine_problems_mean_field or aggregate is aggregate_solutions_interactions:
        refinement_aggregate = aggregate_solutions_linear
    result = refine_result(
        original,
        solve=solve,
        subproblems=subs,
        refinement=refinement,
        aggregate=refinement_aggregate,
        block_size=block_size,
    )
    qubo.solutions = result.solutions
    qubo.refinement_history = result.refinement_history
    return qubo


@configuration.configured
def qsplit_sampler_refined_iterative(
    qubo: QUBO,
    *,
    refinement: Callable[[list[QUBO], QUBO], list[QUBO]] | None = None,
) -> QUBO:
    loops = int(configuration.get("REFINEMENT_LOOPS", "0"))
    if loops <= 0:
        return qsplit_sampler_iterative(qubo)
    refinement = select_refinement(refinement, refine_problems_linear)
    aggregate = aggregate_solutions_linear_bp if BP else aggregate_solutions_linear
    return _qsplit_sampler_refined(qubo, split_problem_linear, aggregate, refinement)


@configuration.configured
def qsplit_sampler_refined_quadtree(
    qubo: QUBO,
    *,
    refinement: Callable[[list[QUBO], QUBO], list[QUBO]] | None = None,
) -> QUBO:
    loops = int(configuration.get("REFINEMENT_LOOPS", "0"))
    if loops <= 0:
        return qsplit_sampler_quadtree(qubo)
    refinement = select_refinement(refinement, refine_problems_quadtree)
    return _qsplit_sampler_refined(qubo, split_problem_quadtree, aggregate_solutions_quadtree, refinement)


@configuration.configured
def qsplit_sampler_refined_interactions(qubo: QUBO, *, refinement: Callable | None = None) -> QUBO:
    loops = int(configuration.get("REFINEMENT_LOOPS", "0"))
    if loops <= 0:
        return qsplit_sampler_interactions(qubo)
    refinement = select_refinement(refinement, refine_problems_mean_field)
    return _qsplit_sampler_refined(qubo, split_problem_interactions, aggregate_solutions_interactions, refinement)


@configuration.configured
def qsplit_sampler_refined_graph_partitioning(qubo: QUBO, *, refinement: Callable | None = None) -> QUBO:
    loops = int(configuration.get("REFINEMENT_LOOPS", "0"))
    if loops <= 0:
        return qsplit_sampler_graph_partitioning(qubo)
    refinement = select_refinement(refinement, refine_problems_mean_field)
    return _qsplit_sampler_refined(qubo, split_problem_recursive_graph, aggregate_solutions_recursive_graph, refinement)


@configuration.configured
def qsplit_sampler_refined_recursive(qubo: QUBO, *, refinement: Callable | None = None) -> QUBO:
    if int(configuration.get("REFINEMENT_LOOPS", "0")) <= 0:
        return qsplit_sampler_recursive(qubo)
    original = deepcopy(qubo)
    refinement = select_refinement(refinement, refine_problems_mean_field)
    leaves = []
    original.solutions = _sample_recursive(qubo, leaves).solutions.copy(deep=True)
    result = refine_result(original, solve=solve, subproblems=leaves, refinement=refinement)
    qubo.solutions = result.solutions
    qubo.refinement_history = result.refinement_history
    return qubo


@configuration.configured
def qsplit_sampler(
    qubo: QUBO,
    *,
    split: Callable,
    aggregate: Callable,
    refinement: Callable | None = None,
    refinement_aggregate: Callable = aggregate_solutions_linear,
) -> QUBO:
    loops = int(configuration.get("REFINEMENT_LOOPS", "0"))
    original = deepcopy(qubo) if loops > 0 else None
    subs = split(qubo)
    if loops > 0:
        solve_problems(subs, solve)
    else:
        for sub in subs:
            sub.solutions = dummy_solve(sub) if is_empty(sub) else solve(sub)
    result = aggregate(subs, qubo)
    if loops <= 0:
        return result
    original.solutions = result.solutions.copy(deep=True)
    refined = refine_result(
        original,
        solve=solve,
        subproblems=subs,
        refinement=refinement,
        aggregate=refinement_aggregate,
    )
    result.solutions = refined.solutions
    result.refinement_history = refined.refinement_history
    return result
