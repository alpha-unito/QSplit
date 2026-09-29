# Acknowledgement:
# Parts of this code are adapted from the official IBM Quantum documentation
# regarding the Quantum Approximate Optimization Algorithm (QAOA).
# Source: https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm
# Modifications have been made to tailor the implementation to local requirements.

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.passmanager import BasePassManager
from qiskit.primitives import BackendSamplerV2, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.exceptions import TranspilerError
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import SPSA
from qiskit_ibm_runtime import IBMBackend
from qiskit_ibm_runtime import SamplerV2 as RuntimeSamplerV2

from qsplit.adapters.ibm.util import get_variables_mapping
from qsplit.adapters.training import training_subproblem
from qsplit.qubo import QUBO


def __from_qubo_matrix_to_circuit(qubo: QUBO) -> tuple[QuantumCircuit, SparsePauliOp, dict[int, int], list[int]]:
    var_to_qubit, all_vars = get_variables_mapping(qubo)
    num_qubits = len(all_vars)

    pauli_list = []

    for i, row_var in enumerate(qubo.rows_idx):
        for j, col_var in enumerate(qubo.cols_idx):
            coeff = qubo.mat[i, j]
            if coeff == 0:
                continue

            if row_var == col_var:
                pauli_list.append(("Z", [var_to_qubit[row_var]], coeff))
            else:
                pauli_list.append(("ZZ", [var_to_qubit[row_var], var_to_qubit[col_var]], coeff))

    if qubo.offset != 0:
        pauli_list.append(("I" * num_qubits, list(range(num_qubits)), qubo.offset))

    cost_hamiltonian = SparsePauliOp.from_sparse_list(pauli_list, num_qubits)
    cost_hamiltonian = cost_hamiltonian.simplify()

    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2)

    return circuit, cost_hamiltonian, var_to_qubit, all_vars


def _is_ibm_backend(backend) -> bool:
    return isinstance(backend, IBMBackend)


def _is_aer_backend(backend) -> bool:
    return isinstance(backend, AerSimulator)


def __optimize_circuit(candidate_circuit: QuantumCircuit, cost_hamiltonian: SparsePauliOp) -> np.ndarray:
    init_params = [np.pi / 2, np.pi / 2, np.pi, np.pi]
    if candidate_circuit.num_parameters < len(init_params):
        return np.array(init_params)
    candidate_circuit = candidate_circuit.decompose(reps=10)
    estimator = StatevectorEstimator()

    def objective_function(params: list[float]) -> float:
        return __cost_func_estimator(params, candidate_circuit, cost_hamiltonian, estimator)

    return SPSA().minimize(fun=objective_function, x0=init_params).x


def __cost_func_estimator(
    params: list[float], ansatz: QuantumCircuit, hamiltonian: SparsePauliOp, estimator: object
) -> float:
    layout = getattr(ansatz, "layout", None)
    isa_hamiltonian = hamiltonian.apply_layout(layout) if layout is not None else hamiltonian
    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])
    results = job.result()[0]
    cost = results.data.evs
    return cost


def get_qaoa_circuit_optimized(
    backend,
    pm: BasePassManager,
    qubo: QUBO,
) -> tuple[QuantumCircuit, dict[int, int], list[int]]:
    training_qubo = training_subproblem(qubo)
    training_circuit, cost_hamiltonian, _, _ = __from_qubo_matrix_to_circuit(training_qubo)
    params = __optimize_circuit(training_circuit, cost_hamiltonian)
    circuit, _, var_to_qubit, all_vars = __from_qubo_matrix_to_circuit(qubo)
    optimized_logical = circuit.assign_parameters(dict(zip(circuit.parameters, params)))
    optimized_logical.measure_all()
    try:
        optimized_circ = pm.run(optimized_logical)
    except TranspilerError as exc:
        if not (_is_aer_backend(backend) and "not in Target" in str(exc)):
            raise
        optimized_circ = optimized_logical.decompose(reps=10)
    if _is_aer_backend(backend) and any(str(inst.operation.name).lower() == "qaoa" for inst in optimized_circ.data):
        optimized_circ = optimized_circ.decompose(reps=10)
    measured_circ = optimized_circ.copy()
    if measured_circ.num_clbits == 0:
        measured_circ.measure_all()
    return measured_circ, var_to_qubit, all_vars


def run_quantum_optimizer(backend, optimized_circuit: QuantumCircuit) -> dict[int, int]:
    if _is_ibm_backend(backend) and RuntimeSamplerV2 is not None:
        sampler = RuntimeSamplerV2(mode=backend)
    else:
        sampler = BackendSamplerV2(backend=backend)
    if _is_ibm_backend(backend) and hasattr(sampler, "options"):
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.num_randomizations = "auto"
    pub = (optimized_circuit,)
    job = sampler.run([pub], shots=500)
    data_bin = job.result()[0].data
    keys_method = getattr(data_bin, "keys", None)
    available_keys = list(keys_method()) if callable(keys_method) else []
    if hasattr(data_bin, "meas") and hasattr(data_bin.meas, "get_int_counts"):
        return data_bin.meas.get_int_counts()
    if hasattr(data_bin, "c") and hasattr(data_bin.c, "get_int_counts"):
        return data_bin.c.get_int_counts()
    if available_keys:
        for key in available_keys:
            reg = getattr(data_bin, key, None)
            if reg is not None and hasattr(reg, "get_int_counts"):
                return reg.get_int_counts()
    raise RuntimeError("No readable classical register found in SamplerV2 result")
