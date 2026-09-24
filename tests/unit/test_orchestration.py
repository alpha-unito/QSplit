"""Scheduling, retries and job lifecycle without contacting SLURM or a QPU."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from streamflow.quantum import qmetrics
from streamflow.quantum.plugin.connector import helpers, slurm_retry
from streamflow.quantum.plugin.connector import iqm_wms_control as wms


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, ["dwave", "ibm", "iqm"]),
        (" IBM, auto, dwave ", ["ibm", "dwave"]),
        (["IQM", ""], ["iqm"]),
        ("auto", ["dwave", "ibm", "iqm"]),
    ],
)
def test_provider_pool(raw, expected):
    assert helpers.parseProviderPool(raw) == expected


@pytest.mark.parametrize(
    "inflight,queues,usage,expected",
    [
        ({"a": 1}, {"a": 0, "b": 20}, {}, "b"),
        ({}, {"a": 3, "b": 1}, {}, "b"),
        ({}, {"a": 1, "b": 1}, {"a": 2}, "b"),
    ],
)
def test_scheduler_prioritizes_capacity_then_queue_then_usage(inflight, queues, usage, expected):
    selected, cursor = helpers.pickLeastLoadedProvider(
        ["a", "b"], lambda p: (True, queues[p]), lambda _: True, inflight, usage, 0
    )
    assert (selected, cursor) == (expected, 0)


def test_scheduler_round_robin_and_unavailable_fallback():
    for cursor in range(4):
        selected, updated = helpers.pickLeastLoadedProvider(
            ["a", "b"], lambda _: (True, 0), lambda _: True, {}, {}, cursor
        )
        assert selected == ["a", "b"][cursor % 2] and updated == cursor + 1
    assert helpers.pickLeastLoadedProvider(["iqm"], lambda _: (False, 99), lambda _: True, {}, {}, 7) == ("dwave", 7)
    fetch = Mock(side_effect=AssertionError("At-capacity provider must not be probed"))
    assert helpers.pickLeastLoadedProvider(["dwave"], fetch, lambda _: False, {}, {}, 0) == ("dwave", 0)


@pytest.mark.parametrize("value,expected", [(None, 0), ("invalid", 0), (-1, 0), ("3", 3)])
def test_queue_clamping(value, expected):
    assert helpers.clampQueueLength(value) == expected


def test_provider_slot_waits_and_releases_without_leaking():
    async def scenario():
        condition, inflight = asyncio.Condition(), {}
        assert helpers.providerHasCapacity(None, None, inflight, "dwave")
        await helpers.acquireProviderSlot("dwave", 5, {"dwave": 1}, inflight, condition)
        assert not helpers.providerHasCapacity(5, {"dwave": 1}, inflight, "dwave")
        waiting = asyncio.create_task(helpers.acquireProviderSlot("dwave", 1, None, inflight, condition))
        await asyncio.sleep(0)
        assert not waiting.done()
        await helpers.releaseProviderSlot("dwave", 1, None, inflight, condition)
        await asyncio.wait_for(waiting, timeout=1)
        assert inflight == {"dwave": 1}
        await helpers.releaseProviderSlot("dwave", 1, None, inflight, condition)
        assert inflight == {}
        await helpers.acquireProviderSlot("ibm", None, None, inflight, condition)
        await helpers.releaseProviderSlot("ibm", None, None, inflight, condition)
        assert inflight == {}

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "responses,services,expected",
    [
        ([("ok", 0)], ["cpu"], ("ok", 0)),
        ([("ordinary failure", 1)], ["cpu"], ("ordinary failure", 1)),
        ([(b"*** JOB 1 CANCELLED AT now DUE TO limit", 143), (b"ok", 0)], ["cpu", "cpu"], ("ok", 0)),
        (
            [("*** JOB 1 CANCELLED AT now DUE TO limit", 0)] * 4,
            ["cpu", "cpu", "fallback", "fallback"],
            ("*** JOB 1 CANCELLED AT now DUE TO limit", 143),
        ),
    ],
)
def test_slurm_retry_and_service_failover(responses, services, expected):
    async def scenario():
        seen, pending = [], iter(responses)

        async def locate(index, service):
            seen.append(service)
            return service

        async def run(location):
            return next(pending)

        result = await slurm_retry.run_with_slurm_cancellation_retry(
            logger=logging.getLogger(__name__),
            service_candidates=["cpu", "fallback"],
            get_location_for_service=locate,
            run_on_location=run,
            retries=1,
            retry_delay=0,
        )
        assert seen == services
        assert result == expected

    asyncio.run(scenario())


def test_simulator_metrics_never_probe_qpu(monkeypatch):
    monkeypatch.setattr(qmetrics, "get_iqm_quantum_backend", Mock(side_effect=AssertionError("QPU call")))
    for provider in ["dwave", "ibm", "cudaq"]:
        assert helpers.fetchProviderStateFor(provider, [], lambda _: True) == (True, 0)
    assert helpers.fetchProviderStateFor("auto", ["dwave", "ibm"], lambda _: True) == (True, 0)
    assert helpers.fetchProviderStateFor("auto", ["dwave"], lambda _: False) == (False, 0)


def test_iqm_probe_failure_is_unavailable(monkeypatch):
    monkeypatch.setattr(qmetrics, "get_iqm_quantum_backend", Mock(side_effect=RuntimeError("offline")))
    assert helpers.fetchProviderStateFor("iqm", ["iqm"], lambda _: True) == (False, 2**31 - 1)


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"status": {"queue_length": " +3 "}}, 3),
        ({"jobs_in_queue": 0}, 0),
        ([{"pending": 4}], 4),
        ({"queue": True}, None),
        ({"depth": -2}, 0),
        ({"queue": 1.5}, None),
        ({"queue": "bad"}, None),
    ],
)
def test_metrics_queue_parsing(payload, expected):
    assert qmetrics._extract_queue_length(payload) == expected


@pytest.mark.parametrize(
    "payload,active",
    [({"healthy": False}, False), ({"state": "maintenance"}, False), ({"status": "ready"}, True), ({}, True)],
)
def test_metrics_health_parsing(payload, active):
    assert qmetrics._extract_active(payload) is active


def test_iqm_metrics_with_fake_client_and_architecture_fallback():
    client = SimpleNamespace(
        get_health=lambda: {"healthy": False, "queue_size": 2},
        get_about=lambda: {"name": "Example Device"},
        get_dynamic_quantum_architecture=Mock(side_effect=RuntimeError("unsupported")),
        get_static_quantum_architecture=lambda: SimpleNamespace(qubits=["q1", "q2"], dut_label="Test QPU"),
    )
    assert qmetrics.iqm_qpu_metrics(client) == {
        "name": "test_qpu",
        "active": False,
        "qubits": 2,
        "fidelity": 0.0,
        "queue": 2,
    }


def test_registry_filters_corruption_deduplicates_and_respects_pid(tmp_path):
    env = {"QSPLIT_IQM_STATE_DIR": str(tmp_path)}
    wms._write_active_registry([{"job_id": "foreign", "pid": -1}, {"job_id": "", "pid": 0}, "invalid"], env)
    wms.add_active_iqm_job("local", env)
    wms.add_active_iqm_job("local", env)
    assert wms.get_active_iqm_job_ids(env) == ["local"]
    assert wms.get_active_iqm_job_ids(env, include_all_pids=True) == ["foreign", "local"]
    wms.remove_active_iqm_job("foreign", env)
    assert "foreign" in wms.get_active_iqm_job_ids(env, include_all_pids=True)
    wms.remove_active_iqm_job("foreign", env, include_all_pids=True)
    assert wms.get_active_iqm_job_ids(env, include_all_pids=True) == ["local"]
    wms.reset_iqm_runtime_state(env)
    assert wms.get_active_iqm_job_ids(env) == []
    wms._active_jobs_path(env).write_text("broken JSON")
    assert wms.get_active_iqm_job_ids(env) == []


@pytest.mark.parametrize("fail", [False, True])
def test_job_result_detaches_even_on_failure(monkeypatch, fail):
    monkeypatch.setattr(wms, "_LOCAL_ACTIVE_JOBS", {})
    monkeypatch.setenv("QSPLIT_IQM_FALLBACK_TIMEOUT_SEC", "10")
    result = Mock(side_effect=TimeoutError() if fail else None, return_value="done")
    job = SimpleNamespace(job_id=lambda: "test-job", result=result)
    wms._attach_job(job)
    assert wms.get_active_iqm_job_ids() == ["test-job"]
    if fail:
        with pytest.raises(TimeoutError):
            job.result()
    else:
        assert job.result() == "done"
    result.assert_called_once_with(timeout=10, cancel_after_timeout=True)
    assert wms._LOCAL_ACTIVE_JOBS == {} and wms.get_active_iqm_job_ids() == []


def test_cleanup_cancels_fake_jobs_and_clears_registry(monkeypatch):
    job = SimpleNamespace(cancel=Mock())
    monkeypatch.setattr(wms, "_LOCAL_ACTIVE_JOBS", {"test-job": job})
    cancel = Mock(return_value=1)
    monkeypatch.setattr(wms, "cancel_iqm_job_ids", cancel)
    wms.add_active_iqm_job("test-job")
    assert wms.cleanup_active_iqm_jobs("test") == 1
    job.cancel.assert_called_once()
    cancel.assert_called_once_with(["test-job"], None)
    assert wms._LOCAL_ACTIVE_JOBS == {} and not wms.get_active_iqm_job_ids()
    assert not wms._CLEANUP_RUNNING


@pytest.mark.parametrize("raw,expected", [("", None), ("-1", None), ("no", None), ("15", 15)])
def test_timeout_configuration(raw, expected):
    assert wms.resolve_iqm_timeout_seconds({"QSPLIT_IQM_FALLBACK_TIMEOUT_SEC": raw}) == expected
