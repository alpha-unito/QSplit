import logging

import dimod
import dwave.system

from adapters.dummy import solve as dummy_solve
from adapters.dwave_sa import solve as sa_solve
from qsplit.qubo import QUBO
from qsplit.aggregate import aggregate_solutions
from qsplit.split import split_problem

log = logging.getLogger('subqubo')


class QSplitSampler:
    def __init__(self, sampler: dimod.SimulatedAnnealingSampler | dwave.system.EmbeddingComposite, cut_dim: int):
        self.sampler = sampler
        self.cut_dim = cut_dim

    def run(self, qubo: QUBO, dim: int) -> QUBO:
        if dim <= self.cut_dim:
            if qubo.problem_size == 0:
                qubo.solutions = dummy_solve(qubo)
            else:
                qubo.solutions = sa_solve(qubo)
            return qubo

        sub_problems = split_problem(qubo)
        solutions = [self.run(p, dim // 2) for p in sub_problems]
        return aggregate_solutions(solutions, qubo)
