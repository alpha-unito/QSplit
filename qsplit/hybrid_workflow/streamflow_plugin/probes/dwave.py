from __future__ import annotations

import asyncio
from typing import Optional

from dwave.cloud import Client

from ..connectors.quantum_availability import BackendHealth
from .base import BaseProbe


class DWaveProbe(BaseProbe):
    def __init__(
        self,
        token: Optional[str] = None,
        endpoint: Optional[str] = None,
        solver_filters: Optional[dict] = None,
    ) -> None:
        self._token = token
        self._endpoint = endpoint
        self._solver_filters = solver_filters or {}

    def _client(self) -> Client:
        if self._token:
            kwargs = {"token": self._token}
            if self._endpoint:
                kwargs["endpoint"] = self._endpoint
            return Client(**kwargs)
        return Client.from_config()

    async def health(self, backend: str) -> BackendHealth:
        def _sync_call() -> BackendHealth:
            with self._client() as c:
                if backend:
                    solver = c.get_solver(name=backend)
                else:
                    # fallback: “un QPU solver disponibile”
                    solver = c.get_solver(qpu=True, **self._solver_filters)

                online = bool(getattr(solver, "online", False))
                avg_load = getattr(solver, "avg_load", None)

                return BackendHealth(
                    provider="dwave",
                    backend=str(getattr(solver, "id", backend or "<default>")),
                    healthy=online,
                    avg_load=float(avg_load) if avg_load is not None else None,
                    status_msg="online" if online else "offline",
                    raw={
                        "online": online,
                        "avg_load": avg_load,
                    },
                )

        return await asyncio.to_thread(_sync_call)