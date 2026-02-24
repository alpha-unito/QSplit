import json
import os
from collections.abc import Mapping, Sequence
from enum import Enum, auto
from math import pi, sqrt
from typing import Any

IQM_DEFAULT_SERVER_URL = "https://resonance.meetiqm.com/"

_ACTIVE_STATES = {"healthy", "ok", "up", "ready", "active", "online", "operational"}
_INACTIVE_STATES = {"unhealthy", "down", "offline", "error", "maintenance", "disabled", "inactive"}
_QUEUE_KEYS = (
    "queue_length",
    "pending_jobs",
    "jobs_in_queue",
    "queued_jobs",
    "pending",
    "waiting",
    "queue_size",
    "depth",
    "queue",
)


def _as_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        if value.is_integer():
            return max(0, int(value))
        return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.startswith("+"):
            raw = raw[1:]
        if raw.isdigit():
            return int(raw)
    return None


def _extract_queue_length(payload: Any, depth: int = 0) -> int | None:
    if depth > 4:
        return None
    direct = _as_non_negative_int(payload)
    if direct is not None:
        return direct
    if isinstance(payload, Mapping):
        for key in _QUEUE_KEYS:
            if key in payload:
                value = _extract_queue_length(payload[key], depth + 1)
                if value is not None:
                    return value
        for value in payload.values():
            if isinstance(value, (Mapping, list, tuple)):
                nested = _extract_queue_length(value, depth + 1)
                if nested is not None:
                    return nested
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            nested = _extract_queue_length(item, depth + 1)
            if nested is not None:
                return nested
    return None


def _extract_active(*payloads: Any) -> bool:
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        for key in ("healthy", "active", "operational", "online", "ready"):
            value = payload.get(key)
            if isinstance(value, bool):
                return value
        status = payload.get("status") or payload.get("state")
        if isinstance(status, str):
            normalized = status.strip().lower()
            if normalized in _ACTIVE_STATES:
                return True
            if normalized in _INACTIVE_STATES:
                return False
    return True


def _resolve_iqm_auth() -> str:
    token = os.getenv("IQM_TOKEN", "").strip() or os.getenv("QSPLIT_IQM_TOKEN", "").strip()
    if not token:
        raise RuntimeError("IQM auth is missing: set IQM_TOKEN.")
    return token


def _resolve_iqm_url() -> str:
    return (
        os.getenv("IQM_SERVER_URL", "").strip()
        or os.getenv("QSPLIT_IQM_SERVER_URL", "").strip()
        or IQM_DEFAULT_SERVER_URL
    )


def _resolve_iqm_quantum_computer() -> str | None:
    qc = os.getenv("IQM_QUANTUM_COMPUTER", "").strip() or os.getenv("QSPLIT_IQM_QUANTUM_COMPUTER", "").strip()
    return qc or None


def get_ibm_quantum_backend():
    from qiskit_ibm_runtime import QiskitRuntimeService

    backend = QiskitRuntimeService(
        channel="ibm_cloud",
        token=os.environ["TOKEN_IBM"],
        instance=os.environ["CRN_IBM"],
    ).least_busy()
    return get_quantum_metrics(backend, BackendType.IBM_QPU)


def get_ibm_classical_backend():
    from qiskit_aer import AerSimulator

    backend = AerSimulator(method="matrix_product_state")
    return backend


def get_iqm_quantum_backend():
    from iqm.iqm_client import IQMClient

    _resolve_iqm_auth()
    iqm_url = _resolve_iqm_url()
    quantum_computer = _resolve_iqm_quantum_computer()

    auth_kwargs: dict[str, str] = {}
    if quantum_computer:
        auth_kwargs["quantum_computer"] = quantum_computer

    return IQMClient(iqm_url, **auth_kwargs)


def iqm_qpu_metrics(client):
    health: dict[str, Any] = {}
    about: dict[str, Any] = {}
    try:
        fetched_health = client.get_health()
        if isinstance(fetched_health, dict):
            health = fetched_health
    except Exception:
        health = {}
    try:
        fetched_about = client.get_about()
        if isinstance(fetched_about, dict):
            about = fetched_about
    except Exception:
        about = {}

    active = _extract_active(health, about)

    queue_length = _extract_queue_length(health)
    if queue_length is None:
        queue_length = _extract_queue_length(about)
    if queue_length is None:
        queue_length = 0

    name_candidates: list[str | None] = [
        str(about.get("name")).strip() if "name" in about else None,
        str(about.get("quantum_computer")).strip() if "quantum_computer" in about else None,
        getattr(client, "quantum_computer", None),
        _resolve_iqm_quantum_computer(),
    ]

    qubits = 0
    try:
        dqa = client.get_dynamic_quantum_architecture()
        qubits = len(list(getattr(dqa, "qubits", []) or []))
    except Exception:
        qubits = 0
    if qubits <= 0:
        try:
            sqa = client.get_static_quantum_architecture()
            qubits = len(list(getattr(sqa, "qubits", []) or []))
            name_candidates.insert(0, getattr(sqa, "dut_label", None))
        except Exception:
            qubits = 0

    name = next((str(candidate).strip() for candidate in name_candidates if candidate and str(candidate).strip()), "")
    if not name:
        name = "iqm_qpu"

    return {
        "name": name.lower().replace(" ", "_"),
        "active": bool(active),
        "qubits": qubits,
        "fidelity": 0.0,
        "queue": queue_length,
    }


def get_dwave_quantum_backend():
    from dwave.system import DWaveSampler, EmbeddingComposite

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
    return {
        "name": backend.name.lower().replace(" ", "_"),
        "active": True,
        "qubits": backend.num_qubits,
        "fidelity": 1.0,
        "queue": 0,
    }


def ibm_qpu_metrics(backend):
    props = backend.properties()
    assert props
    two_q_errors = [gate.parameters[0].value for gate in props.gates if len(gate.qubits) == 2]
    avg_two_q = sum(two_q_errors) / len(two_q_errors) if two_q_errors else 0.0
    readout_errors = [props.readout_error(q_idx) for q_idx in range(backend.num_qubits)]
    avg_readout = sum(readout_errors) / len(readout_errors) if readout_errors else 0.0
    fidelity = 1.0 - ((avg_readout / 2) + (avg_two_q / 2))
    return {
        "name": backend.name.lower().replace(" ", "_"),
        "active": backend.status().operational,
        "qubits": backend.num_qubits,
        "fidelity": fidelity,
        "queue": getattr(backend.status(), "pending_jobs"),
    }


def dwave_qpu_metrics(backend):
    """
    WARNING: this is not tested due to D-Wave access policy

    The number of qubits is estimated with the following proportion: sqrt(qpu_qubits*qpu_average_degree/PI)
    """
    name = backend.solver.id
    n_phys = backend.properties["num_qubits"]
    num_couplers = len(backend.properties["couplers"])
    avg_degree = (2 * num_couplers) / n_phys

    return {
        "name": name.lower().replace(" ", "_"),
        "active": True,
        "qubits": int(sqrt((n_phys * avg_degree) / pi)),
        "fidelity": 1.0,
        "queue": len(backend.client.get_jobs(status="pending", solver=name)),
    }


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
    provider = (os.getenv("QSPLIT_QMETRICS_PROVIDER", "iqm") or "iqm").strip().lower()

    if provider == "iqm":
        backend = get_iqm_quantum_backend()
        metrics = get_quantum_metrics(backend, BackendType.IQM_QPU)
    elif provider in {"ibm", "ibm_gpu"}:
        metrics = get_ibm_quantum_backend()
    elif provider == "dwave":
        backend = get_dwave_quantum_backend()
        metrics = get_quantum_metrics(backend, BackendType.DWAVE_QPU)
    else:
        raise RuntimeError(f"Unsupported provider '{provider}'. Use: iqm, ibm, dwave.")

    print(json.dumps(metrics, indent=2, sort_keys=True))


"""
Quantinuum calculate cost:

HQC = 5 + (N1 + 10 * N2 + 5 * Nm)*C / 5000

N1: Number of one qubit operations
N2: Number of two qubit operations
Nm: Number of measurements
C: Number of shots
"""
