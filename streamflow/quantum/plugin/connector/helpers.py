from __future__ import annotations

from typing import MutableSequence


def parseProviderPool(
    provider_pool: MutableSequence[str] | str | None,
    default_pool: list[str] | None = None,
) -> list[str]:
    if default_pool is None:
        default_pool = ["dwave", "ibm", "iqm"]
    if provider_pool is None:
        return list(default_pool)
    if isinstance(provider_pool, str):
        items = [str(p).strip().lower() for p in provider_pool.split(",")]
    else:
        items = [str(p).strip().lower() for p in provider_pool]
    pool = [p for p in items if p and p != "auto"]
    return pool or list(default_pool)


def pickLeastLoadedProvider(
    provider_pool: list[str],
    fetch_state,
    has_capacity,
    inflight: dict[str, int],
    usage: dict[str, int],
    round_robin_index: int,
    fallback: str = "dwave",
) -> tuple[str, int]:
    candidates: list[tuple[int, int, int, str]] = []
    for candidate in provider_pool:
        if not has_capacity(candidate):
            continue
        active, queue = fetch_state(candidate)
        if not active:
            continue
        candidates.append(
            (
                int(inflight.get(candidate, 0)),
                int(clampQueueLength(queue)),
                int(usage.get(candidate, 0)),
                candidate,
            )
        )
    if not candidates:
        return fallback, round_robin_index

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    best_key = candidates[0][:3]
    tied = [provider for i, q, u, provider in candidates if (i, q, u) == best_key]
    if len(tied) == 1:
        return tied[0], round_robin_index

    idx = round_robin_index % len(tied)
    return tied[idx], round_robin_index + 1


def clampQueueLength(value: int | str | None) -> int:
    try:
        queue_length = int(value)
    except Exception:
        queue_length = 0
    return max(0, queue_length)


def providerHasCapacity(
    max_provider_jobs: int | None,
    provider_max_jobs: dict[str, int] | None,
    provider_inflight: dict[str, int],
    provider: str,
) -> bool:
    limit = max_provider_jobs
    if provider_max_jobs:
        override = provider_max_jobs.get(provider)
        if override is not None:
            limit = override
    if limit is None:
        return True
    return provider_inflight.get(provider, 0) < limit


async def acquireProviderSlot(
    provider: str,
    max_provider_jobs: int | None,
    provider_max_jobs: dict[str, int] | None,
    provider_inflight: dict[str, int],
    condition,
) -> None:
    limit = max_provider_jobs
    if provider_max_jobs:
        override = provider_max_jobs.get(provider)
        if override is not None:
            limit = override
    if limit is None:
        return
    async with condition:
        while provider_inflight.get(provider, 0) >= limit:
            await condition.wait()
        provider_inflight[provider] = provider_inflight.get(provider, 0) + 1


async def releaseProviderSlot(
    provider: str,
    max_provider_jobs: int | None,
    provider_max_jobs: dict[str, int] | None,
    provider_inflight: dict[str, int],
    condition,
) -> None:
    limit = max_provider_jobs
    if provider_max_jobs:
        override = provider_max_jobs.get(provider)
        if override is not None:
            limit = override
    if limit is None:
        return
    async with condition:
        current = provider_inflight.get(provider, 0)
        if current <= 1:
            provider_inflight.pop(provider, None)
        else:
            provider_inflight[provider] = current - 1
        condition.notify_all()


def fetchProviderStateFor(
    provider: str | None,
    provider_pool: list[str],
    provider_state_cache,
    has_capacity,
) -> tuple[bool, int]:
    metrics_active = True
    metrics_queue = 0
    normalized = str(provider or "").strip().lower()
    try:
        from streamflow.quantum import qmetrics

        if normalized != "auto":
            if normalized in provider_state_cache:
                return provider_state_cache[normalized]
        metrics = None
        if normalized == "auto":
            best_queue: int | None = None
            any_active = False
            for candidate in provider_pool:
                if not has_capacity(candidate):
                    continue
                active, queue = fetchProviderStateFor(candidate, provider_pool, provider_state_cache, has_capacity)
                if not active:
                    continue
                any_active = True
                if best_queue is None or queue < best_queue:
                    best_queue = queue
            if any_active:
                metrics_active = True
                metrics_queue = best_queue if best_queue is not None else 0
            else:
                metrics_active = False
                metrics_queue = 0
        else:
            match normalized:
                case "ibm" | "ibm_gpu":
                    metrics = qmetrics.get_ibm_quantum_backend()
                case "dwave":
                    backend = qmetrics.get_dwave_quantum_backend()
                    metrics = qmetrics.get_quantum_metrics(backend, qmetrics.BackendType.DWAVE_QPU)
                case "iqm":
                    backend = qmetrics.get_iqm_quantum_backend()
                    metrics = qmetrics.get_quantum_metrics(backend, qmetrics.BackendType.IQM_QPU)
                case _:
                    metrics = None
            if isinstance(metrics, dict):
                metrics_active = bool(metrics.get("active", True))
                metrics_queue = metrics.get("queue", 0)
    except Exception:
        metrics_active = True
        metrics_queue = 0
    active = metrics_active
    queue_length = clampQueueLength(metrics_queue)
    if normalized != "auto":
        provider_state_cache[normalized] = (active, queue_length)
    return active, queue_length
