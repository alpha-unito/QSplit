from __future__ import annotations

import asyncio
from typing import Optional

from qiskit_ibm_runtime import QiskitRuntimeService

from ..connectors.quantum_availability import BackendHealth
from .base import BaseProbe


class IBMProbe(BaseProbe):
    def __init__(
        self,
        token: Optional[str] = None,
        instance: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> None:
        self._token = token
        self._instance = instance
        self._channel = channel

    def _service(self) -> QiskitRuntimeService:
        # se hai già salvato l’account localmente, token può essere None
        if self._token:
            kwargs = {"token": self._token}
            if self._instance:
                kwargs["instance"] = self._instance
            if self._channel:
                kwargs["channel"] = self._channel
            return QiskitRuntimeService(**kwargs)
        return QiskitRuntimeService()

    async def health(self, backend: str) -> BackendHealth:
        def _sync_call() -> BackendHealth:
            service = self._service()
            b = service.backend(backend) if backend else service.least_busy(operational=True, simulator=False)
            st = b.status()
            return BackendHealth(
                provider="ibm",
                backend=b.name,
                healthy=bool(st.operational),
                pending_jobs=int(st.pending_jobs) if st.pending_jobs is not None else None,
                status_msg=str(st.status_msg) if getattr(st, "status_msg", None) is not None else None,
                raw={
                    "operational": bool(st.operational),
                    "pending_jobs": getattr(st, "pending_jobs", None),
                    "status_msg": getattr(st, "status_msg", None),
                },
            )

        return await asyncio.to_thread(_sync_call)