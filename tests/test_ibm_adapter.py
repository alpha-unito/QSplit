# Copyright (C) 2025  The QSplit Contributors.
# See the 'CONTRIBUTORS' file at the top-level directory of this distribution.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import unittest

import numpy as np
import pandas as pd
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp

from qsplit.adapters.ibm.ibm_qaoa_cpu_noiseless import solve as cpu_solve
from qsplit.adapters.ibm.util import __from_qubo_matrix_to_circuit as from_qubo_matrix_to_circuit
from qsplit.adapters.ibm.util import __get_variables_mapping as get_variables_mapping
from qsplit.adapters.ibm.util import to_dataframe
from qsplit.qubo import QUBO


class TestIBMAdapter(unittest.TestCase):
    ##################################################
    # __get_variables_mapping                        #
    ##################################################

    def test_overlapping_indices(self):
        qubo = QUBO(mat=np.triu(np.ones((2, 2))), rows_idx=np.array([1, 2]), cols_idx=np.array([2, 3]))
        expected_vars = [1, 2, 3]
        expected_mapping = {1: 0, 2: 1, 3: 2}
        var_to_qubit, all_vars = get_variables_mapping(qubo)

        self.assertEqual(all_vars, expected_vars)
        self.assertEqual(var_to_qubit, expected_mapping)
        self.assertEqual(len(all_vars), 3)

    def test_disjoint_indices(self):
        qubo = QUBO(mat=np.triu(np.ones((2, 2))), rows_idx=np.array([10, 11]), cols_idx=np.array([20, 21]))
        expected_vars = [10, 11, 20, 21]
        expected_mapping = {10: 0, 11: 1, 20: 2, 21: 3}
        var_to_qubit, all_vars = get_variables_mapping(qubo)

        self.assertEqual(all_vars, expected_vars)
        self.assertEqual(var_to_qubit, expected_mapping)
        self.assertEqual(len(all_vars), 4)

    def test_single_index_set(self):
        qubo = QUBO(np.triu(np.ones((3, 3))), rows_idx=np.array([0, 1, 2]), cols_idx=np.array([0, 1, 2]))
        expected_vars = [-1, 0, 1, 2]
        expected_mapping = {-1: 0, 0: 1, 1: 2, 2: 3}
        var_to_qubit, all_vars = get_variables_mapping(qubo)

        self.assertEqual(all_vars, expected_vars)
        self.assertEqual(var_to_qubit, expected_mapping)
        self.assertEqual(len(all_vars), 4)

    ##################################################
    # __from_qubo_matrix_to_circuit                  #
    ##################################################

    def test_standard_qubo_mapping(self):
        mat = np.array([[0, 1], [0, 0]])
        qubo = QUBO(mat=mat, rows_idx=np.array([0, 1]), cols_idx=np.array([0, 1]))
        circuit, hamiltonian, var_to_qubit, all_vars = from_qubo_matrix_to_circuit(qubo)

        self.assertEqual(all_vars, [0, 1])
        self.assertEqual(var_to_qubit, {0: 0, 1: 1})
        self.assertEqual(len(all_vars), 2)
        expected_hamiltonian = SparsePauliOp.from_sparse_list([("ZZ", [0, 1], 1.0)], len(all_vars))
        self.assertTrue(expected_hamiltonian.equiv(hamiltonian))
        self.assertIsInstance(circuit, QAOAAnsatz)
        self.assertEqual(circuit.num_qubits, 2)
        self.assertTrue(circuit.clbits)

    def test_bipartite_mapping(self):
        mat = np.array([[1.5, 0.0], [0.0, 2.0]])
        qubo = QUBO(mat=mat, rows_idx=np.array([10, 12]), cols_idx=np.array([20, 21]))
        circuit, hamiltonian, var_to_qubit, all_vars = from_qubo_matrix_to_circuit(qubo)

        self.assertEqual(all_vars, [10, 12, 20, 21])
        self.assertEqual(var_to_qubit, {10: 0, 12: 1, 20: 2, 21: 3})
        self.assertEqual(len(all_vars), 4)
        expected_hamiltonian = SparsePauliOp.from_sparse_list([("ZZ", [0, 2], 1.5), ("ZZ", [1, 3], 2.0)], len(all_vars))
        self.assertTrue(expected_hamiltonian.equiv(hamiltonian))
        self.assertEqual(circuit.num_qubits, 4)

    def test_zero_terms_ignored(self):
        mat = np.array([[0, 10.0], [0, 0]])
        qubo = QUBO(mat=mat, rows_idx=np.array([10, 11]), cols_idx=np.array([10, 11]))
        circuit, hamiltonian, var_to_qubit, all_vars = from_qubo_matrix_to_circuit(qubo)
        expected_hamiltonian = SparsePauliOp.from_sparse_list(
            [
                ("ZZ", [0, 1], 10.0),
            ],
            len(all_vars),
        )

        self.assertTrue(expected_hamiltonian.equiv(hamiltonian))
        self.assertEqual(len(hamiltonian.paulis), 1)

    ##################################################
    # to_dataframe                                   #
    ##################################################

    def test_to_dataframe_filtering_min_energy(self):
        mat = np.array([[1.0, 0.0], [0.0, 1.0]])
        rows = np.array([1, 2])
        cols = np.array([1, 2])
        qubo = QUBO(mat, rows, cols)
        counts_int = {0: 50, 1: 20, 2: 20, 3: 10}
        all_vars = [1, 2]
        var_to_qubit = {1: 0, 2: 1}
        df = to_dataframe(counts_int, qubo, var_to_qubit, all_vars)

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["energy"], 0.0)
        self.assertEqual(df.iloc[0][1], 0)
        self.assertEqual(df.iloc[0][2], 0)

    def test_to_dataframe_variable_mapping_and_ordering(self):
        mat = np.array([[10.0, 0.0], [0.0, 1.0]])
        rows = np.array([10, 20])
        cols = np.array([10, 20])
        qubo = QUBO(mat, rows, cols)
        counts_int = {1: 100, 2: 100}
        all_vars = [10, 20]
        var_to_qubit = {10: 0, 20: 1}

        df = to_dataframe(counts_int, qubo, var_to_qubit, all_vars)
        self.assertEqual(len(df), 1)
        best_row = df.iloc[0]
        self.assertEqual(best_row["energy"], 1.0)
        self.assertEqual(best_row[10], 0)
        self.assertEqual(best_row[20], 1)

    def test_to_dataframe_with_padding_variable(self):
        mat_raw = np.array([[5.0]])
        rows_raw = np.array([1])
        cols_raw = np.array([1])
        qubo = QUBO(mat_raw, rows_raw, cols_raw)

        self.assertIn(-1, qubo.rows_idx)
        all_vars = [1, -1]
        var_to_qubit = {1: 0, -1: 1}
        counts_int = {0: 50, 1: 50}
        df = to_dataframe(counts_int, qubo, var_to_qubit, all_vars)

        self.assertEqual(df.iloc[0]["energy"], 0.0)
        self.assertEqual(df.iloc[0][1], 0)
        self.assertIn(-1, df.columns)

    ##################################################
    # cpu_noiseless                                  #
    ##################################################

    def test_ibm_cpu_solve(self):
        rows_idx = np.array([1, 2])
        cols_idx = np.array([1, 2])
        mat = np.array([[0, 1], [0, 0]])
        qubo = QUBO(mat=mat, rows_idx=rows_idx, cols_idx=cols_idx)
        expected_dataframe = pd.DataFrame({1: [1, 0, 0], 2: [0, 1, 0], "energy": [0, 0, 0]}).reset_index(drop=True)
        actual_dataframe = cpu_solve(qubo).reset_index(drop=True)
        self.assertEqual(expected_dataframe["energy"].min(), actual_dataframe["energy"].min())
        var_cols = [col for col in expected_dataframe.columns if col not in ["energy"]]
        expected_solutions = set(expected_dataframe[var_cols].apply(tuple, axis=1))
        actual_solutions = set(actual_dataframe[var_cols].apply(tuple, axis=1))
        self.assertTrue(all(x in expected_solutions for x in actual_solutions))


if __name__ == "__main__":
    unittest.main()
