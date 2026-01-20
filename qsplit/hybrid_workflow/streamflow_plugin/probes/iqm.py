from __future__ import annotations

import asyncio
from typing import Optional

from iqm.iqm_client import IQMClient

from ..connectors.quantum_availability import BackendHealth
from .base import BaseProbe


class IQMProbe(BaseProbe):
    def __init__(
        self,
        iqm_server_url: str,
        token: Optional[str] = None,
        tokens_file: Optional[str] = None,
        quantum_computer: Optional[str] = None,
    ) -> None:
        self._url = iqm_server_url
        self._token = token
        self._tokens_file = tokens_file
        self._qc = quantum_computer

    def _client(self) -> IQMClient:
        return IQMClient(
            self._url,
            quantum_computer=self._qc,
            token=self._token,
            tokens_file=self._tokens_file,
        )

    async def health(self, backend: str) -> BackendHealth:
        def _sync_call() -> BackendHealth:
            c = self._client()
            h = c.get_health()
            # il formato può variare per deployment IQM; teniamolo “raw” senza assumere campi rigidi
            return BackendHealth(
                provider="iqm",
                backend=backend or (self._qc or "<default>"),
                healthy=True,
                status_msg="ok",
                raw={"health": h},
            )

        return await asyncio.to_thread(_sync_call)
