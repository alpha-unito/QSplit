import unittest

import dimod
import numpy as np
import pandas as pd

from qsplit.adapters.dwave.dwave_sa import solve as solve_sa
from qsplit.adapters.dwave.util import from_qubo_matrix_to_bqm, to_dataframe
from qsplit.qubo import QUBO


class TestDWaveAdapter(unittest.TestCase):
    def setUp(self):
        self.mat = np.array([[1.0, -2.0], [0.0, 1.0]], dtype=np.float64)
        self.rows_idx = np.array([0, 1], dtype=int)
        self.cols_idx = np.array([0, 1], dtype=int)
        self.offset = 5.0
        self.qubo = QUBO(self.mat, self.rows_idx, self.cols_idx, offset=self.offset)

    def test_conversion_matrix_to_bqm(self):
        bqm = from_qubo_matrix_to_bqm(self.qubo)
        self.assertIsInstance(bqm, dimod.BinaryQuadraticModel)
        self.assertEqual(bqm.vartype, dimod.BINARY)
        self.assertEqual(bqm.linear[0], 1.0)
        self.assertEqual(bqm.linear[1], 1.0)
        self.assertEqual(bqm.quadratic[(0, 1)], -2.0)
        self.assertEqual(bqm.offset, 5.0)

    def test_to_dataframe(self):
        samples = np.array([[0, 0], [1, 1], [0, 1], [0, 0]])
        energies = np.array([1.5, 5.0, 1.5, 1.5])
        sampleset = dimod.SampleSet.from_samples(samples, vartype=dimod.BINARY, energy=energies)
        df = to_dataframe(sampleset, self.qubo)

        self.assertNotIn('num_occurrences', df.columns)
        self.assertEqual(len(df), 2)
        self.assertTrue((df['energy'] == 1.5).all())

    def test_solve_simulated_annealing(self):
        df_result = solve_sa(self.qubo)
        self.assertIsInstance(df_result, pd.DataFrame)
        self.assertFalse(df_result.empty)
        self.assertIn('energy', df_result.columns)
        self.assertIn(0, df_result.columns)
        self.assertIn(1, df_result.columns)


if __name__ == '__main__':
    unittest.main()
