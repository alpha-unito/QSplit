from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import MutableSequence

logger = logging.getLogger(__name__)
_PROBED_QPU_PROVIDERS = {"iqm"}


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
    provider_state: dict[str, tuple[str, int | None]] = {}
    candidates: list[tuple[int, int, int, str]] = []
    for candidate in provider_pool:
        if not has_capacity(candidate):
            provider_state[candidate] = ("capacity", None)
            continue
        active, queue = fetch_state(candidate)
        queue_length = int(clampQueueLength(queue))
        provider_state[candidate] = ("active" if active else "inactive", queue_length)
        if not active:
            continue
        candidates.append(
            (
                int(inflight.get(candidate, 0)),
                queue_length,
                int(usage.get(candidate, 0)),
                candidate,
            )
        )
    if not candidates:
        if len(provider_pool) == 1 and provider_pool[0] == fallback:
            only_state = provider_state.get(fallback)
            if only_state is not None and only_state[0] == "capacity":
                logger.info(
                    "Provider '%s' is at maxConcurrentJobs; waiting for the next free slot.",
                    fallback,
                )
            elif only_state is not None and only_state[0] == "inactive":
                logger.warning(
                    "Provider '%s' is unavailable (active=false, queue=%s).",
                    fallback,
                    only_state[1],
                )
            return fallback, round_robin_index
        iqm_state = provider_state.get("iqm")
        if iqm_state is not None and iqm_state[0] == "inactive":
            logger.warning(
                "IQM is unavailable (active=false, queue=%s). Falling back to provider '%s'.",
                iqm_state[1],
                fallback,
            )
        elif iqm_state is not None and iqm_state[0] == "capacity":
            logger.warning(
                "IQM has no available capacity. Falling back to provider '%s'.",
                fallback,
            )
        return fallback, round_robin_index

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    best_key = candidates[0][:3]
    tied = [provider for i, q, u, provider in candidates if (i, q, u) == best_key]
    selected_provider = tied[0] if len(tied) == 1 else tied[round_robin_index % len(tied)]
    iqm_state = provider_state.get("iqm")
    if selected_provider != "iqm" and iqm_state is not None:
        if iqm_state[0] == "inactive":
            logger.warning(
                "Routing to provider '%s' because IQM is unavailable (active=false, queue=%s).",
                selected_provider,
                iqm_state[1],
            )
        elif iqm_state[0] == "capacity":
            logger.warning(
                "Routing to provider '%s' because IQM has no available capacity.",
                selected_provider,
            )
    if len(tied) == 1:
        return selected_provider, round_robin_index

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
    has_capacity,
    provider_env_map: dict[str, dict[str, str]] | None = None,
) -> tuple[bool, int]:
    metrics_active = True
    metrics_queue = 0
    normalized = str(provider or "").strip().lower()
    try:
        if normalized not in {"auto"} and normalized not in _PROBED_QPU_PROVIDERS:
            active = True
            queue_length = 0
            return active, queue_length
        from streamflow.quantum import qmetrics

        metrics = None
        if normalized == "auto":
            best_queue: int | None = None
            any_active = False
            for candidate in provider_pool:
                if not has_capacity(candidate):
                    continue
                active, queue = fetchProviderStateFor(
                    candidate,
                    provider_pool,
                    has_capacity,
                    provider_env_map,
                )
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
                case "iqm":
                    try:
                        backend = qmetrics.get_iqm_quantum_backend()
                        metrics = qmetrics.get_quantum_metrics(backend, qmetrics.BackendType.IQM_QPU)
                    except ModuleNotFoundError as missing_exc:
                        if not str(getattr(missing_exc, "name", "") or "").startswith("iqm"):
                            raise
                        metrics = _probe_iqm_metrics_via_subprocess(provider_env_map)
                case _:
                    metrics = None
            if isinstance(metrics, dict):
                metrics_active = bool(metrics.get("active", True))
                metrics_queue = metrics.get("queue", 0)
                if normalized == "iqm":
                    logger.info(
                        "IQM traffic probe (live): active=%s queue=%s qubits=%s name=%s",
                        metrics_active,
                        metrics_queue,
                        metrics.get("qubits"),
                        metrics.get("name"),
                    )
    except Exception as exc:
        if normalized == "iqm":
            logger.warning("Failed IQM provider state probe: %s", exc)
            metrics_active = False
            metrics_queue = 2**31 - 1
        else:
            metrics_active = True
            metrics_queue = 0
    active = metrics_active
    queue_length = clampQueueLength(metrics_queue)
    return active, queue_length


def _probe_iqm_metrics_via_subprocess(
    provider_env_map: dict[str, dict[str, str]] | None,
) -> dict | None:
    env = dict(os.environ)
    iqm_env = (provider_env_map or {}).get("iqm", {})
    env.update(iqm_env)
    env["QSPLIT_QMETRICS_PROVIDER"] = "iqm"
    python_bin = _resolve_probe_python_bin(iqm_env)
    script = Path(__file__).resolve().parents[2] / "qmetrics.py"
    output = subprocess.run(
        [python_bin, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=20,
    )
    if output.returncode != 0:
        err = (output.stderr or "").strip() or "no stderr"
        raise RuntimeError(f"IQM probe command failed ({output.returncode}): {err}")
    payload = (output.stdout or "").strip()
    if not payload:
        return None
    return json.loads(payload)


def _resolve_probe_python_bin(iqm_env: dict[str, str]) -> str:
    explicit = str(iqm_env.get("PYTHON_BIN", "")).strip()
    if explicit:
        return explicit
    venv_path = str(iqm_env.get("VIRTUAL_ENV", "")).strip()
    if not venv_path:
        venv_path = str(os.getenv("QSPLIT_IQM_VENV", "")).strip()
    if venv_path:
        candidate = Path(venv_path) / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    return "python3"
