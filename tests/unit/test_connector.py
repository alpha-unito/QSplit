import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("streamflow.core.deployment")
from streamflow.core.deployment import ExecutionLocation  # noqa: E402

from streamflow.quantum.plugin.connector import quantum_connector as connector  # noqa: E402

pytestmark = pytest.mark.streamflow


@pytest.fixture
def wrapper(tmp_path, monkeypatch):
    monkeypatch.setattr(connector, "_SHARED_DISABLED_PROVIDERS", {})
    (tmp_path / "provider.yaml").write_text("QSPLIT_BACKEND: dwave\n")
    inner = Mock(run=AsyncMock(return_value=("done", 0)), get_available_locations=AsyncMock(return_value={}))
    return connector.QuantumConnectorWrapper(
        "test",
        str(tmp_path),
        inner,
        provider="dwave",
        providerPool=["dwave", "ibm"],
        providerServiceMap={"dwave": "cpu", "ibm": "cpu2"},
        providerServiceFallbackMap={"dwave": "cpu-backup"},
        providerConfigMap={"dwave": "provider.yaml"},
        maxConcurrentJobs={"dwave": 1, "ibm": 2},
    )


def location():
    inner = ExecutionLocation(name="local", deployment="local", local=True)
    return ExecutionLocation(name="dwave:local", deployment="test", local=True, service="cpu", wraps=inner)


def test_wrapper_schema_and_configuration(wrapper):
    assert json.loads(wrapper.get_schema())["type"] == "object"
    assert wrapper._provider_config_map == {"dwave": {"QSPLIT_BACKEND": "dwave"}}
    assert wrapper._provider_service_fallback_map == {"dwave": ["cpu-backup"]}
    assert wrapper._provider_has_capacity("dwave")
    wrapper._disable_provider("dwave", "test failure")
    assert not wrapper._provider_has_capacity("dwave")
    assert wrapper._pick_auto_provider() == "ibm"


@pytest.mark.parametrize("failure", [False, True])
def test_run_forwards_provider_environment_and_releases_slot(wrapper, failure):
    if failure:
        wrapper.connector.run.side_effect = RuntimeError("command failed")

    async def scenario():
        if failure:
            with pytest.raises(RuntimeError, match="command failed"):
                await wrapper.run(location(), ["cli_scatter"], environment={"ORIGINAL": "yes"})
        else:
            assert await wrapper.run(location(), ["cli_scatter"], environment={"ORIGINAL": "yes"}) == ("done", 0)
        assert wrapper._provider_inflight == {}
        env = wrapper.connector.run.call_args.kwargs["environment"]
        assert env == {"ORIGINAL": "yes"}
        assert wrapper.connector.run.call_args.kwargs["command"] == ["cli_scatter", "--backend", "dwave"]
        assert wrapper._provider_usage == {"dwave": 1}

    asyncio.run(scenario())


def test_no_available_provider_fails_without_running(wrapper):
    wrapper._disable_provider("dwave", "test")
    wrapper._disable_provider("ibm", "test")
    output, status = asyncio.run(wrapper.run(location(), ["cli_scatter"]))
    assert status == 1 and "No provider available" in output
    wrapper.connector.run.assert_not_called()


@pytest.mark.parametrize(
    "command,wrapped", [([], False), (["echo", "hello"], False), (["/env/bin/cli_scatter", "--input-qubo", "x"], True)]
)
def test_iqm_wrapper_only_intercepts_scatter(command, wrapped):
    result = connector.QuantumConnectorWrapper._wrap_iqm_command(command, "/env/python")
    if wrapped:
        assert result == ["/env/python", "-m", "streamflow.quantum.plugin.connector.iqm_scatter_wrapper", *command[1:]]
    else:
        assert result == command


@pytest.mark.parametrize(
    "status,message,expected",
    [(124, "", True), (1, "Internal Server Error", True), (1, "connection timed out", True), (1, "bad input", False)],
)
def test_transient_error_classification(status, message, expected):
    assert connector.QuantumConnectorWrapper._is_iqm_transient_failure(status, message) is expected


def test_non_solver_steps_do_not_receive_backend_flags_or_credentials(wrapper):
    wrapper._provider_config_map["dwave"]["TOKEN_IBM"] = "secret-sentinel"
    asyncio.run(wrapper.run(location(), ["cli_split", "--input-matrix", "input.csv"]))
    kwargs = wrapper.connector.run.call_args.kwargs
    assert kwargs["command"] == ["cli_split", "--input-matrix", "input.csv"]
    assert kwargs["environment"] == {}
    assert "secret-sentinel" not in repr(kwargs)


@pytest.mark.parametrize("probe_status", [0, 75])
def test_iqm_probe_keeps_staged_config_args_and_reserves_only_on_cache_miss(wrapper, monkeypatch, probe_status):
    wrapper._provider_pool = ["iqm"]
    wrapper._provider_config_map = {"iqm": {"IQM_TOKEN": "secret-sentinel"}}
    wrapper._provider_python_map = {"iqm": "/iqm/bin/python"}
    wrapper._provider_service_map = {"iqm": "cpu"}
    wrapper.connector.run.side_effect = [("probe", probe_status), ("solved", 0)]
    acquire, release = AsyncMock(), AsyncMock()
    monkeypatch.setattr(wrapper, "_acquire_provider_slot", acquire)
    monkeypatch.setattr(wrapper, "_release_provider_slot", release)
    loc = location()
    loc.name = "iqm:local"
    command = ["cli_scatter", "--config", "staged-private.yaml", "--input-qubo", "q.pkl"]
    result = asyncio.run(wrapper.run(loc, command))
    assert result == (("probe", 0) if probe_status == 0 else ("solved", 0))
    calls = wrapper.connector.run.call_args_list
    assert len(calls) == (1 if probe_status == 0 else 2)
    for index, call in enumerate(calls):
        args = call.kwargs["command"]
        assert args[:3] == ["/iqm/bin/python", "-m", "streamflow.quantum.plugin.connector.iqm_scatter_wrapper"]
        assert args[3:7] == ["--config", "staged-private.yaml", "--input-qubo", "q.pkl"]
        assert ("--cache-only" in args) == (index == 0)
        assert call.kwargs["environment"] == {}
        assert "secret-sentinel" not in repr(call)
    assert acquire.await_count == release.await_count == (0 if probe_status == 0 else 1)
