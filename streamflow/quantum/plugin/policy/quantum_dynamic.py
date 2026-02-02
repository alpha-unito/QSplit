from __future__ import annotations

from importlib.resources import files
from typing import Any, Dict, MutableMapping, Optional

from streamflow.core.scheduling import Policy

from .helpers import count_running_jobs, get_location_jobs, resolve_provider


class QuantumDynamicPolicy(Policy):

    @classmethod
    def get_schema(cls) -> str:
        return (
            files("streamflow.quantum.plugin")
            .joinpath("schemas")
            .joinpath("quantum_dynamic_policy.json")
            .read_text("utf-8")
        )

    def __init__(self, **config: Any) -> None:
        self._target_provider: Dict[str, str] = config.get("targetProvider", {})
        self._fallback_cost: float = float(config.get("fallbackCost", 1e9))
        try:
            self._usage_penalty: float = float(config.get("usagePenalty", 0.0))
        except Exception:
            self._usage_penalty = 0.0

        max_jobs = config.get("maxConcurrentJobs")
        if max_jobs in (None, 0, "0"):
            self._max_concurrent: Optional[int] = None
        else:
            try:
                self._max_concurrent = int(max_jobs)
            except Exception:
                self._max_concurrent = None
        self._provider_usage: Dict[str, int] = {}

    async def get_location(
        self,
        context: Any,
        job: Any,
        hardware_requirement: Any,
        available_locations: MutableMapping[str, Any],
        jobs: MutableMapping[str, Any],
        locations: MutableMapping[str, MutableMapping[str, Any]],
    ) -> Any:
        best_location: Optional[Any] = None
        best_cost: Optional[float] = None
        best_provider: Optional[str] = None

        def _alloc_jobs(loc_alloc: Any) -> list[str]:
            for attr in ("jobs", "job_names", "jobNames"):
                if hasattr(loc_alloc, attr):
                    try:
                        return list(getattr(loc_alloc, attr))
                    except Exception:
                        break
            if isinstance(loc_alloc, dict):
                try:
                    return list(loc_alloc.get("jobs") or [])
                except Exception:
                    return []
            return []

        provider_running_jobs: Dict[str, int] = {
            provider: 0 for provider in set(self._target_provider.values())
        }
        for deployment_name, loc_allocs in locations.items():
            for location_name, loc_alloc in loc_allocs.items():
                provider = (
                    self._target_provider.get(location_name)
                    or self._target_provider.get(deployment_name)
                )
                if provider is None:
                    continue
                provider_running_jobs[provider] = (
                    provider_running_jobs.get(provider, 0)
                    + count_running_jobs(_alloc_jobs(loc_alloc), jobs)
                )

        providers_in_scope: set[str] = set()
        for loc_name, loc in available_locations.items():
            provider = resolve_provider(self._target_provider, loc_name, loc)
            if provider is not None:
                providers_in_scope.add(provider)
        location_running_jobs: Dict[str, int] = {}
        if self._max_concurrent is not None:
            for loc_name, loc in available_locations.items():
                location_running_jobs[loc_name] = count_running_jobs(
                    get_location_jobs(loc_name, loc, locations), jobs
                )
        allowed_providers = providers_in_scope

        for loc_name, loc in available_locations.items():
            provider = resolve_provider(self._target_provider, loc_name, loc)
            job_count = (
                location_running_jobs.get(loc_name, 0)
                if self._max_concurrent is not None
                else 0
            )
            if provider is None:
                if self._max_concurrent is not None and job_count >= self._max_concurrent:
                    continue
                cost = float(self._fallback_cost)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_location = loc
                continue
            if provider not in allowed_providers:
                continue

            running_jobs = provider_running_jobs.get(provider, 0)

            capacity_penalty = 0.0
            if self._max_concurrent is not None and job_count >= self._max_concurrent:
                capacity_penalty = float(job_count - self._max_concurrent + 1)

            usage_penalty = self._usage_penalty * float(
                self._provider_usage.get(provider, 0)
            )
            cost = float(running_jobs) + capacity_penalty + usage_penalty
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_location = loc
                best_provider = provider

        if best_location is not None:
            if best_provider:
                self._provider_usage[best_provider] = (
                    self._provider_usage.get(best_provider, 0) + 1
                )

        return best_location
