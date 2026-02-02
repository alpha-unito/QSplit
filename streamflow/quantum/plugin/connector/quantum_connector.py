from __future__ import annotations

import asyncio
import logging
from importlib.resources import files
from typing import MutableMapping, MutableSequence, Optional
from cachetools import TTLCache
from streamflow.core.asyncache import cachedmethod
from streamflow.core.deployment import Connector, ExecutionLocation
from streamflow.core.scheduling import AvailableLocation
from streamflow.deployment.wrapper import ConnectorWrapper
from streamflow.log_handler import logger


class QuantumConnectorWrapper(ConnectorWrapper):

    def __init__(
        self,
        deployment_name: str,
        config_dir: str,
        connector: Connector,
        service: str | None = None,
        provider: str = "classic",
        availabilityProbability: Optional[float] = None,
        maxQueueLength: Optional[int] = None,
        cacheTTL: int = 3,
        transferBufferSize: int = 2**16,
    ) -> None:
        super().__init__(
            deployment_name=deployment_name,
            config_dir=config_dir,
            connector=connector,
            service=service,
            transferBufferSize=transferBufferSize,
        )
        self._provider = (provider or "classic").strip().lower()
        self._availability_probability: Optional[float] = None
        if availabilityProbability not in (None, ""):
            try:
                self._availability_probability = float(availabilityProbability)
            except Exception:
                self._availability_probability = None
        if self._availability_probability is not None:
            if self._availability_probability < 0:
                self._availability_probability = 0.0
            elif self._availability_probability > 1:
                self._availability_probability = 1.0
        if maxQueueLength in (None, ""):
            self._max_queue = None
        else:
            try:
                self._max_queue = int(maxQueueLength)
            except Exception:
                self._max_queue = None
        try:
            cache_ttl = max(0, int(cacheTTL))
        except Exception:
            cache_ttl = 3
        self._locations_cache: TTLCache = TTLCache(maxsize=32, ttl=cache_ttl)

    @classmethod
    def get_schema(cls) -> str:
        return (
            files("streamflow.quantum.plugin")
            .joinpath("schemas")
            .joinpath("quantum_connector.json")
            .read_text("utf-8")
        )

    def _is_quantum_provider(self) -> bool:
        return self._provider not in {"classic", "classical", "dummy"}

    def _resolve_service(self, service: str | None, loc_service: str | None) -> str | None:
        return (service or self.service or loc_service)

    def _cache_key(self, service: str | None) -> str:
        key = service or self.service or ""
        return str(key).strip().lower()

    def _inner_location(self, location: ExecutionLocation) -> ExecutionLocation:
        return location.wraps if getattr(location, "wraps", None) is not None else location

    def _inner_locations(
        self, locations: MutableSequence[ExecutionLocation]
    ) -> MutableSequence[ExecutionLocation]:
        return [self._inner_location(loc) for loc in locations]

    def _fetch_provider_state(self) -> tuple[bool, int]:
        metrics_active = True
        metrics_queue = 0
        try:
            from streamflow.quantum.qmetrics import get_ibm_quantum_backend

            metrics = get_ibm_quantum_backend()
            if isinstance(metrics, dict):
                metrics_active = bool(metrics.get("active", True))
                metrics_queue = metrics.get("queue", 0)
        except Exception:
            metrics_active = True
            metrics_queue = 0
        active = metrics_active
        queue_length = metrics_queue
        try:
            queue_length = int(queue_length)
        except Exception:
            queue_length = 0
        if queue_length < 0:
            queue_length = 0
        return active, queue_length

    @cachedmethod(lambda self: self._locations_cache, key=lambda service=None: str(service or "").strip().lower(),)
    async def _get_quantum_locations_cached(self, service: str | None = None) -> MutableMapping[str, AvailableLocation]:
        
        inner_locations = await self.connector.get_available_locations(service=service)
        available, queue_length = self._fetch_provider_state()


        if not available:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "QuantumConnectorWrapper provider %s reported unavailable; skipping locations",
                    self._provider,
                )
            return {}
        if (
            self._availability_probability is not None
            and self._availability_probability <= 0
        ):
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "QuantumConnectorWrapper provider %s availabilityProbability=%s disables locations",
                    self._provider,
                    self._availability_probability,
                )
            return {}
        if self._max_queue is not None and queue_length > self._max_queue:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "QuantumConnectorWrapper provider %s queue %d exceeds max %d, skipping locations",
                    self._provider,
                    queue_length,
                    self._max_queue,
                )
            return {}

        wrapped_locations: MutableMapping[str, AvailableLocation] = {}
        for name, loc in inner_locations.items():
            wrapped = AvailableLocation(
                name=loc.name,
                deployment=self.deployment_name,
                hostname=loc.hostname,
                local=loc.local,
                service=self._resolve_service(service, loc.service),
                slots=loc.slots,
                stacked=True,
                hardware=loc.hardware,
                wraps=loc,
            )
            env = wrapped.location.environment
            env.setdefault("QSPLIT_PROVIDER", self._provider)
            env.setdefault("QSPLIT_QUEUE_LENGTH", str(queue_length))
            env.setdefault("QSPLIT_PROVIDER_AVAILABLE", str(available).lower())
            wrapped_locations[name] = wrapped
        return wrapped_locations

    async def get_available_locations(
        self, service: str | None = None
    ) -> MutableMapping[str, AvailableLocation]:
        if self._is_quantum_provider():
            resolved_service = service or self.service
            cache_key = self._cache_key(resolved_service)
            cache_key in self._locations_cache
            return await self._get_quantum_locations_cached(resolved_service)

        inner_locations = await self.connector.get_available_locations(
            service=service or self.service
        )
        return {
            name: AvailableLocation(
                name=loc.name,
                deployment=self.deployment_name,
                hostname=loc.hostname,
                local=loc.local,
                service=self._resolve_service(service, loc.service),
                slots=loc.slots,
                stacked=True,
                hardware=loc.hardware,
                wraps=loc,
            )
            for name, loc in inner_locations.items()
        }

    async def run(
        self,
        location: ExecutionLocation,
        command: MutableSequence[str],
        environment: MutableMapping[str, str] | None = None,
        workdir: str | None = None,
        stdin: int | str | None = None,
        stdout: int | str = asyncio.subprocess.STDOUT,
        stderr: int | str = asyncio.subprocess.STDOUT,
        capture_output: bool = False,
        timeout: int | None = None,
        job_name: str | None = None,
    ) -> tuple[str, int] | None:
        env = dict(environment or {})
        env.setdefault("QSPLIT_BACKEND", self._provider or "classic")
        return await self.connector.run(
            location=self._inner_location(location),
            command=command,
            environment=env,
            workdir=workdir,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            capture_output=capture_output,
            timeout=timeout,
            job_name=job_name,
        )

    async def copy_local_to_remote(
        self,
        src: str,
        dst: str,
        locations: MutableSequence[ExecutionLocation],
        read_only: bool = False,
    ) -> None:
        await self.connector.copy_local_to_remote(
            src=src,
            dst=dst,
            locations=self._inner_locations(locations),
            read_only=read_only,
        )

    async def copy_remote_to_local(
        self,
        src: str,
        dst: str,
        location: ExecutionLocation,
        read_only: bool = False,
    ) -> None:
        await self.connector.copy_remote_to_local(
            src=src,
            dst=dst,
            location=self._inner_location(location),
            read_only=read_only,
        )

    async def copy_remote_to_remote(
        self,
        src: str,
        dst: str,
        locations: MutableSequence[ExecutionLocation],
        source_location: ExecutionLocation,
        source_connector: Connector | None = None,
        read_only: bool = False,
    ) -> None:
        if source_connector is None:
            source_connector = self
        await self.connector.copy_remote_to_remote(
            src=src,
            dst=dst,
            locations=self._inner_locations(locations),
            source_location=self._inner_location(source_location),
            source_connector=source_connector,
            read_only=read_only,
        )

    async def get_stream_reader(
        self, command: MutableSequence[str], location: ExecutionLocation
    ):
        return await self.connector.get_stream_reader(
            command, self._inner_location(location)
        )

    async def get_stream_writer(
        self, command: MutableSequence[str], location: ExecutionLocation
    ):
        return await self.connector.get_stream_writer(
            command, self._inner_location(location)
        )
