from __future__ import annotations

import asyncio
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Dict, Optional

from streamflow.core.deployment import Connector
from streamflow.core.exception import WorkflowExecutionException

from ..probes.dwave import DWaveProbe
from ..probes.ibm import IBMProbe
from ..probes.iqm import IQMProbe


@dataclass(frozen=True)
class BackendHealth:
    provider: str
    backend: str
    healthy: bool
    pending_jobs: Optional[int] = None
    avg_load: Optional[float] = None
    status_msg: Optional[str] = None
    raw: Optional[dict] = None


class QuantumAvailabilityConnector(Connector):
    """
    Connector "wrapper" che NON esegue job: espone una fonte di informazione runtime
    (health/queue) per decisioni di resource allocation.

    In StreamFlow, i Connector sono istanziati dal DeploymentManager per ogni deployment
    definito in streamflow.yml. Questo connector va usato come deployment dedicato,
    e poi richiamato dal tuo workflow/filters/policy (dipende da come vuoi integrare
    la decisione di scheduling).
    """

    @classmethod
    def get_schema(cls) -> str:
        return (
            files(__package__)
            .joinpath("..")  # connectors/
            .joinpath("schemas")
            .joinpath("quantum_availability.json")
            .read_text("utf-8")
        )

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._config = config

        provider = config.get("provider")
        if provider not in ("ibm", "dwave", "iqm"):
            raise WorkflowExecutionException(f"Invalid provider={provider}. Expected one of: ibm, dwave, iqm")
        self._provider = provider

        # vendor-specific
        self._probe_timeout_s = int(config.get("timeoutSeconds", 10))
        self._backend = config.get("backend", "")

        self._ibm_cfg = config.get("ibm", {})
        self._dwave_cfg = config.get("dwave", {})
        self._iqm_cfg = config.get("iqm", {})

    async def close(self) -> None:
        # niente da chiudere: i client li creiamo “on demand”
        return None

    # ---- API "pubblica" che userai nella tua logica di allocation ----
    async def get_backend_health(self) -> BackendHealth:
        """
        Ritorna uno snapshot di health/queue del backend selezionato in config.
        """
        try:
            return await asyncio.wait_for(self._probe(), timeout=self._probe_timeout_s)
        except asyncio.TimeoutError as e:
            return BackendHealth(
                provider=self._provider,
                backend=self._backend or "<default>",
                healthy=False,
                status_msg=f"probe timeout after {self._probe_timeout_s}s",
                raw={"error": repr(e)},
            )
        except Exception as e:
            return BackendHealth(
                provider=self._provider,
                backend=self._backend or "<default>",
                healthy=False,
                status_msg="probe exception",
                raw={"error": repr(e)},
            )

    async def _probe(self) -> BackendHealth:
        if self._provider == "ibm":
            probe = IBMProbe(**self._ibm_cfg)
            return await probe.health(self._backend)

        if self._provider == "dwave":
            probe = DWaveProbe(**self._dwave_cfg)
            return await probe.health(self._backend)

        probe = IQMProbe(**self._iqm_cfg)
        return await probe.health(self._backend)
