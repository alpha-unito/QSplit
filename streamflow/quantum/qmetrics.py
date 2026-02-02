from enum import Enum, auto
from math import sqrt, pi

TOKEN_IBM = "Pv9tmBSc9kt65jBmawNJdNsAhrkLAMgje-Wktsrvk1iD"
CRN_IBM = "crn:v1:bluemix:public:quantum-computing:us-east:a/b8009a2acdb6407c8d86042f8caf3448:2576aaaa-1af4-4bfd-9f4f-fd12b60d3476::"
TOKEN_IQM = "HBbk3+3OkEV+M7As4QqBTb97kVVG9A2OL8zqkgdwxbgGiHntAb10woAA0a3E+ECn"
SERVER_IQM = "https://resonance.meetiqm.com/"

# SERVER_IQM = "https://cocos.resonance.meetiqm.com/garnet"


def get_ibm_quantum_backend():
    from qiskit_ibm_runtime import QiskitRuntimeService
    backend = QiskitRuntimeService(
        channel="ibm_cloud",
        token=TOKEN_IBM,
        instance=CRN_IBM,
    ).least_busy()
    return get_quantum_metrics(backend, BackendType.IBM_QPU)


def get_ibm_classical_backend():
    from qiskit_aer import AerSimulator
    backend = AerSimulator(method="matrix_product_state")
    return backend


def get_iqm_quantum_backend():
    """
    iqm-client==33.0.3: usa IQMClient.
    Se il server gestisce più QPU puoi passare quantum_computer="garnet".
    """
    from iqm.iqm_client import IQMClient

    # Se ti serve forzare una QPU specifica:
    # backend = IQMClient(SERVER_IQM, quantum_computer="garnet", token=TOKEN_IQM)

    backend = IQMClient(SERVER_IQM, token=TOKEN_IQM)
    return backend


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    return (sum(vals) / len(vals)) if vals else None


def iqm_qpu_metrics(client):
    """
    Stima automatica di qubits/health/fidelity usando:
      - SQA (qubits + connectivity)
      - ObservationFinder (gate fidelities + readout errors)

    Riferimenti API:
      - IQMClient.get_health / get_static_quantum_architecture / get_calibration_quality_metrics  [oai_citation:6‡docs.meetiqm.com](https://docs.meetiqm.com/iqm-client/api/iqm.iqm_client.iqm_client.IQMClient.html)
      - StaticQuantumArchitecture.{dut_label, qubits, connectivity}  [oai_citation:7‡docs.meetiqm.com](https://docs.meetiqm.com/iqm-client/api/iqm.iqm_client.models.StaticQuantumArchitecture.html)
      - ObservationFinder.get_gate_fidelity / get_measure_errors  [oai_citation:8‡docs.meetiqm.com](https://docs.meetiqm.com/iqm-station-control-client/api/iqm.station_control.client.qon.ObservationFinder.html)
    """
    # 1) Info “statiche” QPU
    sqa = client.get_dynamic_quantum_architecture()  # StaticQuantumArchitecture
    qubits = list(getattr(sqa, "qubits", []))
    connectivity = list(getattr(sqa, "connectivity", []))  # lista di tuple tipo ("QB1","QB2",...)

    # name: meglio dut_label se presente
    name = getattr(sqa, "dut_label", None) or "iqm_qpu"

    # 2) Health (robusto a shape diverse)
    health = client.get_health()  # dict[str, Any]
    # spesso è qualcosa tipo {"healthy": True, ...} ma facciamo fallback
    active = bool(
        health.get("healthy")
        if isinstance(health, dict) and "healthy" in health
        else health.get("status") in ("healthy", "ok", "UP")
        if isinstance(health, dict)
        else True
    )

    # 3) Metriche qualità (calibrazione + quality metrics)
    # Ritorna un ObservationFinder con helper comodi  [oai_citation:9‡docs.meetiqm.com](https://docs.meetiqm.com/iqm-client/api/iqm.iqm_client.iqm_client.IQMClient.html)
    qof = client.get_calibration_quality_metrics()

    # Convenzioni/metodi più comuni per queste metriche:
    # - prx fidelity via randomized benchmarking -> "rb"
    # - measure errors via "ssro"
    # - altre gate fidelities spesso via "irb"
    prx_impl = "rb"
    meas_impl = "ssro"
    cz_impl = "irb"

    # prx: media su qubit singoli
    prx_fids = []
    for qb in qubits:
        f = qof.get_gate_fidelity("prx", prx_impl, (qb,))
        if f is not None:
            prx_fids.append(float(f))
    avg_prx = _mean(prx_fids)

    # cz: media su coppie nella connectivity (se ci sono)
    cz_fids = []
    for edge in connectivity:
        locus = tuple(edge)
        if len(locus) == 2:
            f = qof.get_gate_fidelity("cz", cz_impl, locus)
            if f is not None:
                cz_fids.append(float(f))
    avg_cz = _mean(cz_fids)

    # readout: media degli errori (p10, p01), convertita in "readout fidelity"
    readout_errs = []
    for qb in qubits:
        errs = qof.get_measure_errors("measure", meas_impl, (qb,))
        # errs tipicamente è (err0, err1) oppure None
        if errs is not None and len(errs) == 2:
            e0, e1 = errs
            readout_errs.append((float(e0) + float(e1)) / 2.0)

    avg_readout_err = _mean(readout_errs)
    readout_fidelity = (1.0 - avg_readout_err) if avg_readout_err is not None else None

    # Fidelity aggregata: media delle parti disponibili
    fidelity_parts = [v for v in (avg_prx, avg_cz, readout_fidelity) if v is not None]
    fidelity = _mean(fidelity_parts) if fidelity_parts else None

    return {
        "name": str(name).lower().replace(" ", "_"),
        "active": active,
        "qubits": len(qubits),
        "fidelity": fidelity if fidelity is not None else 0.0,  # oppure None se preferisci
        "queue": 0,  # al momento non esposto come "queue length" affidabile via API
    }


def get_dwave_quantum_backend():
    from dwave.system import EmbeddingComposite, DWaveSampler
    return EmbeddingComposite(DWaveSampler())


def get_dwave_classical_backend():
    from dwave.samplers import SimulatedAnnealingSampler
    return SimulatedAnnealingSampler()


class BackendType(Enum):
    IBM_QPU = auto()
    IBM_SIM = auto()
    DWAVE_QPU = auto()
    DWAVE_SIM = auto()
    IQM_QPU = auto()


def ibm_simulator_metrics(backend):
    return {"name": backend.name.lower().replace(" ", "_"), "active": True, "qubits": backend.num_qubits,
            "fidelity": 1.0, "queue": 0}


def ibm_qpu_metrics(backend):
    props = backend.properties()
    assert props
    two_q_errors = [gate.parameters[0].value for gate in props.gates if len(gate.qubits) == 2]
    avg_two_q = sum(two_q_errors) / len(two_q_errors) if two_q_errors else 0.0
    readout_errors = [props.readout_error(q_idx) for q_idx in range(backend.num_qubits)]
    avg_readout = sum(readout_errors) / len(readout_errors) if readout_errors else 0.0
    fidelity = 1.0 - ((avg_readout / 2) + (avg_two_q / 2))
    return {"name": backend.name.lower().replace(" ", "_"), "active": backend.status().operational,
            "qubits": backend.num_qubits, "fidelity": fidelity, "queue": getattr(backend.status(), 'pending_jobs')}


def dwave_qpu_metrics(backend):
    """
    WARNING: this is not tested due to D-Wave access policy

    The number of qubits is estimated with the following proportion: sqrt(qpu_qubits*qpu_average_degree/PI)
    """
    name = backend.solver.id
    n_phys = backend.properties["num_qubits"]
    num_couplers = len(backend.properties["couplers"])
    avg_degree = (2 * num_couplers) / n_phys

    return {"name": name.lower().replace(" ", "_"), "active": True, "qubits": int(sqrt((n_phys * avg_degree) / pi)),
            "fidelity": 1.0, "queue": len(backend.client.get_jobs(status='pending', solver=name))}


def dwave_simulator_metrics(_):
    return {"name": "simulated_annealing", "active": True, "qubits": 150, "fidelity": 1.0, "queue": 0}


# def iqm_qpu_metrics(backend):
#     """
#     num_qubits and fidelity are hardcoded since there is no API to access this kind of information
#     """
#     return {"name": "iqm_", "active": backend.get_health()["healthy"],
#             "qubits": 20, "fidelity": 0.98265, "queue": 0}  # TODO queue


def get_quantum_metrics(backend, backend_type: BackendType):
    if backend_type == BackendType.IBM_QPU:
        return ibm_qpu_metrics(backend)
    elif backend_type == BackendType.IBM_SIM:
        return ibm_simulator_metrics(backend)
    elif backend_type == BackendType.DWAVE_QPU:
        return dwave_qpu_metrics(backend)
    elif backend_type == BackendType.DWAVE_SIM:
        return dwave_simulator_metrics(backend)
    elif backend_type == BackendType.IQM_QPU:
        return iqm_qpu_metrics(backend)
    else:
        assert False, "Unreachable"


if __name__ == "__main__":
    backend2 = get_ibm_quantum_backend()
    print(backend2)

"""
- Come contiamo i qubit di DWave?
    - Nelle QPU reali i numeri sono tra i 4000 e i 5000, ma nella pratica il problema che si codifica è da 150. Facciamo un brutale /30 sapendo che dipende dall'architettura?
    - Per SA invece? Non c'è un vero limite di qubit, però non mi sembra il caso di metterlo a infinito
-
"""
