from __future__ import annotations

import asyncio
import logging
import os
from importlib.resources import files
from typing import MutableMapping, MutableSequence, Optional

from streamflow.core.deployment import Connector, ExecutionLocation
from streamflow.core.scheduling import AvailableLocation
from streamflow.deployment.wrapper import ConnectorWrapper, get_inner_location, get_inner_locations

from .helpers import (
    acquireProviderSlot,
    fetchProviderStateFor,
    parseProviderPool,
    pickLeastLoadedProvider,
    providerHasCapacity,
    releaseProviderSlot,
)
from .iqm_wms_control import (
    cleanup_active_iqm_jobs,
    install_supervisor_cleanup_handlers,
    reset_iqm_runtime_state,
)
from .slurm_retry import run_with_slurm_cancellation_retry

logger = logging.getLogger(__name__)
_SHARED_DISABLED_PROVIDERS: dict[str, set[str]] = {}


class QuantumConnectorWrapper(ConnectorWrapper):
    def __init__(
        self,
        deployment_name: str,
        config_dir: str,
        connector: Connector,
        service: str | None = None,
        provider: str = "dwave",
        providerPool: Optional[MutableSequence[str] | str] = None,
        providerServiceMap: Optional[MutableMapping[str, str]] = None,
        providerServiceFallbackMap: Optional[MutableMapping[str, MutableSequence[str] | str]] = None,
        providerEnvMap: Optional[MutableMapping[str, MutableMapping[str, str]]] = None,
        maxConcurrentJobs: Optional[MutableMapping[str, int]] = None,
        maxProviderJobs: Optional[int] = None,
        transferBufferSize: int = 2**16,
    ) -> None:
        super().__init__(
            deployment_name=deployment_name,
            config_dir=config_dir,
            connector=connector,
            service=service,
            transferBufferSize=transferBufferSize,
        )
        self._default_provider = str(provider or "dwave").strip().lower() or "dwave"
        if self._default_provider == "auto":
            self._default_provider = "dwave"
        env_pool = os.getenv("QSPLIT_PROVIDER_POOL")
        if env_pool:
            providerPool = env_pool
        self._provider_pool = parseProviderPool(
            providerPool,
            default_pool=[self._default_provider, "ibm_gpu", "iqm"],
        )
        if self._default_provider not in self._provider_pool:
            self._provider_pool.insert(0, self._default_provider)
        self._provider_usage: dict[str, int] = {}
        self._provider_service_map: dict[str, str] = {}
        if providerServiceMap:
            for key, value in providerServiceMap.items():
                provider_name = str(key).strip().lower()
                service_name = str(value).strip().lower()
                if provider_name and service_name:
                    self._provider_service_map[provider_name] = service_name
        self._provider_service_fallback_map: dict[str, list[str]] = {}
        if providerServiceFallbackMap:
            for key, value in providerServiceFallbackMap.items():
                provider_name = str(key).strip().lower()
                if not provider_name:
                    continue
                if isinstance(value, str):
                    services = [s.strip().lower() for s in value.split(",")]
                else:
                    services = [str(s).strip().lower() for s in value]
                services = [s for s in services if s]
                if services:
                    self._provider_service_fallback_map[provider_name] = services
        self._service_provider_map: dict[str, str] = {
            service: provider for provider, service in self._provider_service_map.items()
        }
        self._provider_env_map: dict[str, dict[str, str]] = {}
        if providerEnvMap:
            for provider_key, provider_env in providerEnvMap.items():
                provider_name = str(provider_key).strip().lower()
                if not provider_name or not isinstance(provider_env, MutableMapping):
                    continue
                env_map: dict[str, str] = {}
                for env_key, env_value in provider_env.items():
                    if env_value is None:
                        continue
                    env_name = str(env_key).strip()
                    if not env_name:
                        continue
                    raw_value = str(env_value)
                    expanded_value = os.path.expandvars(raw_value)
                    if "${" in expanded_value and expanded_value == raw_value:
                        continue
                    if expanded_value:
                        env_map[env_name] = expanded_value
                if env_map:
                    self._provider_env_map[provider_name] = env_map
        self._provider_max_jobs: dict[str, int] = {}
        if maxConcurrentJobs:
            for key, value in maxConcurrentJobs.items():
                provider_name = str(key).strip().lower()
                try:
                    limit = int(value)
                except Exception:
                    continue
                if provider_name and limit > 0:
                    self._provider_max_jobs[provider_name] = limit
        if maxProviderJobs in (None, ""):
            self._max_provider_jobs = None
        else:
            try:
                self._max_provider_jobs = int(maxProviderJobs)
                if self._max_provider_jobs <= 0:
                    self._max_provider_jobs = None
            except Exception:
                self._max_provider_jobs = None
        self._provider_inflight: dict[str, int] = {}
        self._provider_condition = asyncio.Condition()
        self._provider_round_robin = 0
        self._disabled_providers = _SHARED_DISABLED_PROVIDERS.setdefault(self.deployment_name, set())
        self._prepare_iqm_runtime_state()

    def _prepare_iqm_runtime_state(self) -> None:
        if "iqm" not in self._provider_pool:
            return
        env = self._provider_env_map.get("iqm", {})
        install_supervisor_cleanup_handlers(env=env)
        try:
            cleanup_active_iqm_jobs(
                "connector_init_cleanup",
                env=env,
                include_all_pids=True,
            )
        except Exception as exc:
            logger.warning("IQM init cleanup failed: %s", exc)
        try:
            reset_iqm_runtime_state(env=env)
            logger.warning("IQM WMS state reset for deployment '%s'.", self.deployment_name)
        except Exception as exc:
            logger.warning("IQM state reset failed: %s", exc)

    @staticmethod
    def _is_iqm_transient_failure(return_code: int, output: str) -> bool:
        text = (output or "").strip().lower()
        if "iqm transient failure" in text:
            return True
        if "internal server error" in text:
            return True
        if "timeout" in text or "timed out" in text:
            return True
        if "read timed out" in text or ("connection" in text and "timed out" in text):
            return True
        return return_code in {124, 137, 143, -9}

    def _provider_has_capacity(self, provider: str) -> bool:
        if provider in self._disabled_providers:
            return False
        return providerHasCapacity(
            self._max_provider_jobs,
            self._provider_max_jobs,
            self._provider_inflight,
            provider,
        )

    @classmethod
    def get_schema(cls) -> str:
        return (
            files("streamflow.quantum.plugin").joinpath("schemas").joinpath("quantum_connector.json").read_text("utf-8")
        )

    def _fetch_provider_state_for(self, provider: str | None) -> tuple[bool, int]:
        return fetchProviderStateFor(
            provider,
            self._provider_pool,
            self._provider_has_capacity,
            self._provider_env_map,
        )

    def _pick_auto_provider(self) -> str:
        fallback = self._default_provider
        provider, self._provider_round_robin = pickLeastLoadedProvider(
            self._provider_pool,
            self._fetch_provider_state_for,
            self._provider_has_capacity,
            self._provider_inflight,
            self._provider_usage,
            self._provider_round_robin,
            fallback=fallback,
        )
        return provider

    def _disable_provider(self, provider: str, reason: str) -> None:
        provider = str(provider or "").strip().lower()
        if not provider:
            return
        if provider in self._disabled_providers:
            return
        self._disabled_providers.add(provider)
        logger.warning("Provider '%s' disabled for this workflow run: %s", provider, reason)

    async def _acquire_provider_slot(self, provider: str) -> None:
        await acquireProviderSlot(
            provider,
            self._max_provider_jobs,
            self._provider_max_jobs,
            self._provider_inflight,
            self._provider_condition,
        )

    async def _release_provider_slot(self, provider: str) -> None:
        await releaseProviderSlot(
            provider,
            self._max_provider_jobs,
            self._provider_max_jobs,
            self._provider_inflight,
            self._provider_condition,
        )

    @staticmethod
    def _wrap_iqm_command(command: MutableSequence[str], env: MutableMapping[str, str]) -> MutableSequence[str]:
        if not command:
            return command
        executable = os.path.basename(str(command[0]).strip().lower())
        if executable not in {"cli_scatter", "cli_scatter.exe"}:
            return command
        python_bin = str(env.get("PYTHON_BIN", "")).strip() or "python"
        return [
            python_bin,
            "-m",
            "streamflow.quantum.plugin.connector.iqm_scatter_wrapper",
            *list(command[1:]),
        ]

    async def _get_quantum_locations(
        self,
        service: str | None = None,
        provider: str | None = None,
    ) -> MutableMapping[str, AvailableLocation]:
        inner_locations = await self.connector.get_available_locations(service=service)
        available, _ = self._fetch_provider_state_for(provider)
        if not available:
            return {}
        return inner_locations

    async def get_available_locations(self, service: str | None = None) -> MutableMapping[str, AvailableLocation]:
        resolved_service = service or self.service
        providers = [self._pick_auto_provider()]

        wrapped_locations: MutableMapping[str, AvailableLocation] = {}
        for provider in providers:
            provider_service = self._provider_service_map.get(provider, resolved_service)
            inner_locations = await self._get_quantum_locations(provider_service, provider)
            for name, loc in inner_locations.items():
                location_name = (
                    f"{provider}:{loc.name}" if self._provider_service_map or len(self._provider_pool) > 1 else loc.name
                )
                wrapped_locations[location_name] = AvailableLocation(
                    name=location_name,
                    deployment=self.deployment_name,
                    hostname=loc.hostname,
                    local=loc.local,
                    service=provider_service or loc.service,
                    slots=loc.slots,
                    stacked=True,
                    hardware=loc.hardware,
                    wraps=loc,
                )
        if not wrapped_locations:
            for provider in self._provider_pool:
                if provider in providers:
                    continue
                provider_service = self._provider_service_map.get(provider, resolved_service)
                inner_locations = await self._get_quantum_locations(provider_service, provider)
                for name, loc in inner_locations.items():
                    location_name = (
                        f"{provider}:{loc.name}"
                        if self._provider_service_map or len(self._provider_pool) > 1
                        else loc.name
                    )
                    wrapped_locations[location_name] = AvailableLocation(
                        name=location_name,
                        deployment=self.deployment_name,
                        hostname=loc.hostname,
                        local=loc.local,
                        service=provider_service or loc.service,
                        slots=loc.slots,
                        stacked=True,
                        hardware=loc.hardware,
                        wraps=loc,
                    )
        return wrapped_locations

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
        base_env = dict(environment or {})
        location_name = str(getattr(location, "name", "") or "").strip().lower()
        preferred_backend: str | None = None
        if ":" in location_name:
            prefix = location_name.split(":", 1)[0].strip().lower()
            if prefix in self._provider_pool:
                preferred_backend = prefix
        if not preferred_backend:
            service_name = str(getattr(location, "service", "") or "").strip().lower()
            preferred_backend = self._service_provider_map.get(service_name) or self._pick_auto_provider()
        provider_candidates: list[str] = []
        if preferred_backend:
            provider_candidates.append(preferred_backend)
        if preferred_backend == "iqm":
            provider_candidates = ["iqm"]
        else:
            provider_candidates.extend([p for p in self._provider_pool if p not in provider_candidates])
        if not provider_candidates:
            raise RuntimeError("No provider configured in providerPool.")

        for backend in provider_candidates:
            if backend in self._disabled_providers:
                logger.warning(
                    "Provider '%s' already disabled for this workflow run. Trying fallback provider.",
                    backend,
                )
                continue

            env = dict(base_env)
            env["QSPLIT_BACKEND"] = backend
            if backend in self._provider_env_map:
                env.update(self._provider_env_map[backend])
            command_to_run = command
            if backend == "iqm":
                command_to_run = self._wrap_iqm_command(command, env)
            effective_timeout = timeout
            if backend == "iqm":
                effective_timeout = None
            resolved_service = str(getattr(location, "service", "") or "").strip().lower()
            primary_service = self._provider_service_map.get(backend, resolved_service or self.service)
            fallback_services = self._provider_service_fallback_map.get(backend, [])
            service_candidates = [primary_service] + [
                service for service in fallback_services if service and service != primary_service
            ]
            if backend:
                self._provider_usage[backend] = self._provider_usage.get(backend, 0) + 1
            await self._acquire_provider_slot(backend)
            try:

                async def _resolve_target_location(service_index: int, target_service: str) -> ExecutionLocation | None:
                    if service_index == 0 and backend == preferred_backend:
                        return get_inner_location(location)
                    available_locations = await self.connector.get_available_locations(service=target_service)
                    if not available_locations:
                        logger.warning(
                            "No available locations found for service '%s' (provider '%s').",
                            target_service,
                            backend,
                        )
                        return None
                    selected = next(iter(available_locations.values()))
                    return getattr(selected, "location", selected)

                async def _run_on_location(target_location: ExecutionLocation):
                    return await self.connector.run(
                        location=target_location,
                        command=command_to_run,
                        environment=env,
                        workdir=workdir,
                        stdin=stdin,
                        stdout=stdout,
                        stderr=stderr,
                        capture_output=capture_output,
                        timeout=effective_timeout,
                        job_name=job_name,
                    )

                try:
                    result = await run_with_slurm_cancellation_retry(
                        logger=logger,
                        service_candidates=service_candidates,
                        get_location_for_service=_resolve_target_location,
                        run_on_location=_run_on_location,
                    )
                except Exception as exc:
                    if backend == "iqm" and isinstance(exc, TimeoutError):
                        logger.warning(
                            "IQM command timeout reached (%s s). Failing current QSplit instance.",
                            effective_timeout,
                        )
                        cleanup_active_iqm_jobs(
                            "connector_timeout",
                            env=env,
                            include_all_pids=True,
                        )
                        raise
                    if backend == "iqm" and self._is_iqm_transient_failure(1, str(exc)):
                        logger.warning("IQM internal server error/timeout detected. Failing current QSplit instance.")
                        cleanup_active_iqm_jobs(
                            "connector_transient_failure",
                            env=env,
                            include_all_pids=True,
                        )
                        raise
                    raise
                if (
                    backend == "iqm"
                    and isinstance(result, tuple)
                    and len(result) == 2
                    and isinstance(result[1], int)
                    and result[1] != 0
                ):
                    output = result[0]
                    if isinstance(output, bytes):
                        output = output.decode(errors="replace")
                    if self._is_iqm_transient_failure(result[1], str(output)):
                        logger.warning("IQM internal server error/timeout detected. Failing current QSplit instance.")
                        cleanup_active_iqm_jobs(
                            "connector_transient_failure",
                            env=env,
                            include_all_pids=True,
                        )
                        return result
                return result
            finally:
                await self._release_provider_slot(backend)

        message = (
            f"No provider available for command (pool={','.join(self._provider_pool)}). "
            "Check maxConcurrentJobs/availability."
        )
        logger.warning(message)
        return message, 1

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
            locations=get_inner_locations(locations),
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
            location=get_inner_location(location),
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
            locations=get_inner_locations(locations),
            source_location=get_inner_location(source_location),
            source_connector=source_connector,
            read_only=read_only,
        )

    async def get_stream_reader(self, command: MutableSequence[str], location: ExecutionLocation):
        return await self.connector.get_stream_reader(command, get_inner_location(location))

    async def get_stream_writer(self, command: MutableSequence[str], location: ExecutionLocation):
        return await self.connector.get_stream_writer(command, get_inner_location(location))
