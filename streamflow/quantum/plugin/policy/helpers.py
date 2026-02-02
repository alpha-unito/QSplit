from __future__ import annotations

from typing import Any, MutableMapping, Optional

from streamflow.core.workflow import Status

_RUNNING_STATUSES = {Status.RUNNING, Status.FIREABLE}


def get_deployment_name(loc: Any) -> Optional[str]:
    for attr in ("deployment_name", "deployment", "deploymentId", "target"):
        if hasattr(loc, attr):
            try:
                deployment_name = getattr(loc, attr)
                if hasattr(deployment_name, "name"):
                    deployment_name = deployment_name.name
                if isinstance(deployment_name, str):
                    return deployment_name
            except Exception:
                continue
    return None


def resolve_provider(
    target_provider: dict[str, str], loc_name: str, loc: Any
) -> Optional[str]:
    provider = target_provider.get(loc_name)
    if provider is not None:
        return provider
    deployment_name = get_deployment_name(loc)
    if deployment_name is not None:
        return target_provider.get(deployment_name)
    return None


def get_location_jobs(
    loc_name: str,
    loc: Any,
    locations: MutableMapping[str, MutableMapping[str, Any]],
) -> list[str]:
    loc_alloc: Optional[Any] = None
    deployment_key = get_deployment_name(loc)
    if isinstance(deployment_key, str):
        dep_allocs = locations.get(deployment_key) or {}
        loc_alloc = dep_allocs.get(loc_name)
        if loc_alloc is None and ":" in loc_name:
            short_name = loc_name.split(":", 1)[1]
            loc_alloc = dep_allocs.get(short_name)
        if loc_alloc is None and hasattr(loc, "name"):
            try:
                loc_alloc = dep_allocs.get(loc.name)
            except Exception:
                loc_alloc = None
    else:
        for dep_allocs in locations.values():
            if loc_name in dep_allocs:
                loc_alloc = dep_allocs[loc_name]
                break
            if loc_alloc is None and ":" in loc_name:
                short_name = loc_name.split(":", 1)[1]
                if short_name in dep_allocs:
                    loc_alloc = dep_allocs[short_name]
                    break

    jobs_list: Optional[Any] = None
    if loc_alloc is not None:
        for attr in ("jobs", "job_names", "jobNames"):
            if hasattr(loc_alloc, attr):
                try:
                    jobs_list = getattr(loc_alloc, attr)
                    break
                except Exception:
                    continue
        if jobs_list is None and isinstance(loc_alloc, dict):
            jobs_list = loc_alloc.get("jobs")
    if jobs_list is None:
        return []
    try:
        return list(jobs_list)
    except Exception:
        return []


def count_running_jobs(job_names: list[str], jobs: MutableMapping[str, Any]) -> int:
    running = 0
    for job_name in job_names:
        job_alloc = jobs.get(job_name)
        if job_alloc is None:
            continue
        status = getattr(job_alloc, "status", None)
        if status in _RUNNING_STATUSES:
            running += 1
    return running
