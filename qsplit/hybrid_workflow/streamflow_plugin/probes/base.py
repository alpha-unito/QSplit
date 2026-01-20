from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..connectors.quantum_availability import BackendHealth


class BaseProbe(ABC):
    @abstractmethod
    async def health(self, backend: str) -> BackendHealth:
        raise NotImplementedError