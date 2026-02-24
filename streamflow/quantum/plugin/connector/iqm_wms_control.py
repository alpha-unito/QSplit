from __future__ import annotations

import atexit
import json
import logging
import os
import signal
from pathlib import Path
from typing import Mapping
from uuid import UUID

logger = logging.getLogger(__name__)

_LOCAL_ACTIVE_JOBS: dict[str, object] = {}
_PATCHED_BACKEND_RUN = False
_SIGNAL_HANDLERS_INSTALLED = False
_SUPERVISOR_HANDLERS_INSTALLED = False
_SUPERVISOR_ENV: dict[str, str] = {}
_PREV_SUPERVISOR_HANDLERS: dict[int, object] = {}
_CLEANUP_RUNNING = False

try:
    import fcntl  # type: ignore[attr-defined]
except Exception:
    fcntl = None


def _env_value(env: Mapping[str, str] | None, key: str) -> str:
    if env is not None:
        value = str(env.get(key, "")).strip()
        if value:
            return value
    return os.getenv(key, "").strip()


def _iqm_state_dir(env: Mapping[str, str] | None = None) -> Path:
    raw = _env_value(env, "QSPLIT_IQM_STATE_DIR")
    if raw:
        return Path(raw).expanduser()
    return Path("/tmp") / "qsplit" / "iqm_state"


def _active_jobs_path(env: Mapping[str, str] | None = None) -> Path:
    return _iqm_state_dir(env) / "iqm_active_jobs.json"


def _lock(file_obj) -> None:
    if fcntl is None:
        return
    fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)


def _unlock(file_obj) -> None:
    if fcntl is None:
        return
    fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)


def _read_json_file(path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as f:
        _lock(f)
        try:
            f.seek(0)
            raw = f.read().strip()
            if not raw:
                return {}
            try:
                data = json.loads(raw)
            except Exception:
                return {}
            if not isinstance(data, dict):
                return {}
            return data
        finally:
            _unlock(f)


def _write_json_file(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as f:
        _lock(f)
        try:
            f.seek(0)
            f.truncate()
            json.dump(data, f, separators=(",", ":"))
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        finally:
            _unlock(f)


def _parse_positive_int(raw: str) -> int | None:
    if not raw:
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    return value if value > 0 else None


def resolve_iqm_timeout_seconds(env: Mapping[str, str] | None = None) -> int | None:
    return _parse_positive_int(_env_value(env, "QSPLIT_IQM_FALLBACK_TIMEOUT_SEC"))


def reset_iqm_runtime_state(env: Mapping[str, str] | None = None) -> None:
    for path in (_active_jobs_path(env),):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            continue


def _read_active_registry(env: Mapping[str, str] | None = None) -> list[dict[str, object]]:
    data = _read_json_file(_active_jobs_path(env))
    entries = data.get("jobs", [])
    if not isinstance(entries, list):
        return []
    out: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        job_id = str(entry.get("job_id", "")).strip()
        if not job_id:
            continue
        try:
            pid = int(entry.get("pid"))
        except Exception:
            continue
        out.append({"job_id": job_id, "pid": pid})
    return out


def _write_active_registry(entries: list[dict[str, object]], env: Mapping[str, str] | None = None) -> None:
    _write_json_file(_active_jobs_path(env), {"jobs": entries})


def add_active_iqm_job(job_id: str, env: Mapping[str, str] | None = None) -> None:
    job_id = str(job_id).strip()
    if not job_id:
        return
    entries = _read_active_registry(env)
    pid = os.getpid()
    if any(str(e.get("job_id")) == job_id and int(e.get("pid", -1)) == pid for e in entries):
        return
    entries.append({"job_id": job_id, "pid": pid})
    _write_active_registry(entries, env)


def remove_active_iqm_job(
    job_id: str,
    env: Mapping[str, str] | None = None,
    *,
    include_all_pids: bool = False,
) -> None:
    job_id = str(job_id).strip()
    if not job_id:
        return
    entries = _read_active_registry(env)
    pid = os.getpid()
    kept = []
    for entry in entries:
        same_job = str(entry.get("job_id", "")).strip() == job_id
        same_pid = int(entry.get("pid", -1)) == pid
        if same_job and (include_all_pids or same_pid):
            continue
        kept.append(entry)
    _write_active_registry(kept, env)


def get_active_iqm_job_ids(
    env: Mapping[str, str] | None = None,
    *,
    include_all_pids: bool = False,
) -> list[str]:
    pid = os.getpid()
    return [
        str(entry["job_id"]) for entry in _read_active_registry(env) if include_all_pids or int(entry["pid"]) == pid
    ]


def _resolve_iqm_auth_env(env: Mapping[str, str] | None = None) -> tuple[str, str, str]:
    url = _env_value(env, "IQM_SERVER_URL")
    token = _env_value(env, "IQM_TOKEN")
    qc = _env_value(env, "IQM_QUANTUM_COMPUTER")
    return url, token, qc


def cancel_iqm_job_ids(job_ids: list[str], env: Mapping[str, str] | None = None) -> int:
    if not job_ids:
        return 0
    url, token, quantum_computer = _resolve_iqm_auth_env(env)
    if not url or not token:
        logger.warning("Cannot cancel IQM jobs: missing IQM_SERVER_URL/IQM_TOKEN.")
        return 0
    try:
        from iqm.iqm_client import IQMClient
    except Exception as exc:
        logger.warning("Cannot cancel IQM jobs: iqm client import failed (%s).", exc)
        return 0

    kwargs: dict[str, str] = {"token": token}
    if quantum_computer:
        kwargs["quantum_computer"] = quantum_computer
    try:
        client = IQMClient(url, **kwargs)
    except Exception as exc:
        logger.warning("Cannot initialize IQM client for cancellation: %s", exc)
        return 0

    cancelled = 0
    for job_id in job_ids:
        try:
            client.cancel_job(UUID(str(job_id)))
            cancelled += 1
        except Exception as exc:
            logger.warning("IQM cancel failed for job_id=%s: %s", job_id, exc)
    return cancelled


def cleanup_active_iqm_jobs(
    reason: str,
    env: Mapping[str, str] | None = None,
    *,
    include_all_pids: bool = False,
) -> int:
    global _CLEANUP_RUNNING
    if _CLEANUP_RUNNING:
        return 0
    _CLEANUP_RUNNING = True
    try:
        tracked = set(get_active_iqm_job_ids(env, include_all_pids=include_all_pids))
        tracked.update(_LOCAL_ACTIVE_JOBS.keys())
        for job_id, job in list(_LOCAL_ACTIVE_JOBS.items()):
            try:
                cancel_method = getattr(job, "cancel", None)
                if callable(cancel_method):
                    cancel_method()
            except Exception:
                pass
            finally:
                _LOCAL_ACTIVE_JOBS.pop(job_id, None)
        cancelled = cancel_iqm_job_ids(sorted(tracked), env)
        for job_id in list(tracked):
            remove_active_iqm_job(job_id, env, include_all_pids=include_all_pids)
        if tracked:
            logger.warning(
                "IQM WMS cleanup (%s): tracked=%s cancelled_via_api=%s",
                reason,
                len(tracked),
                cancelled,
            )
        return cancelled
    finally:
        _CLEANUP_RUNNING = False


def _extract_job_id(job: object) -> str | None:
    raw = getattr(job, "job_id", None)
    if callable(raw):
        try:
            raw = raw()
        except Exception:
            raw = None
    if raw is None:
        raw = getattr(job, "_job_id", None)
    if raw is None:
        return None
    job_id = str(raw).strip()
    return job_id or None


def _detach_job(job_id: str) -> None:
    _LOCAL_ACTIVE_JOBS.pop(job_id, None)
    remove_active_iqm_job(job_id)


def _attach_job(job: object) -> None:
    job_id = _extract_job_id(job)
    if not job_id:
        return
    _LOCAL_ACTIVE_JOBS[job_id] = job
    add_active_iqm_job(job_id)

    result_method = getattr(job, "result", None)
    if not callable(result_method):
        return
    if bool(getattr(job, "_qsplit_wms_result_patched", False)):
        return
    timeout_sec = resolve_iqm_timeout_seconds()

    def _patched_result(*args, **kwargs):
        run_kwargs = dict(kwargs)
        if timeout_sec is not None and "timeout" not in run_kwargs:
            run_kwargs["timeout"] = timeout_sec
        if timeout_sec is not None and "cancel_after_timeout" not in run_kwargs:
            run_kwargs["cancel_after_timeout"] = True
        try:
            return result_method(*args, **run_kwargs)
        finally:
            _detach_job(job_id)

    try:
        setattr(job, "result", _patched_result)
        setattr(job, "_qsplit_wms_result_patched", True)
    except Exception:
        pass


def _runtime_exit_cleanup() -> None:
    cleanup_active_iqm_jobs("runtime_exit", include_all_pids=False)


def _runtime_signal_handler(signum: int, _frame) -> None:
    _runtime_exit_cleanup()
    raise SystemExit(128 + signum)


def _install_runtime_signal_handlers() -> None:
    global _SIGNAL_HANDLERS_INSTALLED
    if _SIGNAL_HANDLERS_INSTALLED:
        return
    atexit.register(_runtime_exit_cleanup)
    for sig_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _runtime_signal_handler)
        except Exception:
            continue
    _SIGNAL_HANDLERS_INSTALLED = True


def install_iqm_runtime_hooks() -> None:
    global _PATCHED_BACKEND_RUN
    if _PATCHED_BACKEND_RUN:
        return
    _install_runtime_signal_handlers()
    try:
        from iqm.qiskit_iqm.iqm_provider import IQMBackend
    except Exception as exc:
        logger.warning("IQM runtime hook install skipped: %s", exc)
        return

    original_run = IQMBackend.run

    def _patched_run(self, *args, **kwargs):
        nonlocal original_run
        job = original_run(self, *args, **kwargs)
        _attach_job(job)
        return job

    IQMBackend.run = _patched_run
    _PATCHED_BACKEND_RUN = True


def _supervisor_exit_cleanup() -> None:
    cleanup_active_iqm_jobs(
        "wms_exit",
        env=_SUPERVISOR_ENV,
        include_all_pids=True,
    )


def _supervisor_signal_handler(signum: int, frame) -> None:
    cleanup_active_iqm_jobs(
        f"wms_signal_{signum}",
        env=_SUPERVISOR_ENV,
        include_all_pids=True,
    )
    prev = _PREV_SUPERVISOR_HANDLERS.get(signum)
    if callable(prev):
        prev(signum, frame)
        return
    if prev == signal.SIG_IGN:
        return
    raise SystemExit(128 + signum)


def install_supervisor_cleanup_handlers(env: Mapping[str, str] | None = None) -> None:
    global _SUPERVISOR_HANDLERS_INSTALLED
    if env:
        _SUPERVISOR_ENV.update({k: str(v) for k, v in env.items()})
    if _SUPERVISOR_HANDLERS_INSTALLED:
        return
    atexit.register(_supervisor_exit_cleanup)
    for sig_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            _PREV_SUPERVISOR_HANDLERS[sig] = signal.getsignal(sig)
            signal.signal(sig, _supervisor_signal_handler)
        except Exception:
            continue
    _SUPERVISOR_HANDLERS_INSTALLED = True
