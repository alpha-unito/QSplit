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
    inner = Mock(run=AsyncMock(return_value=("done", 0)), get_available_locations=AsyncMock(return_value={}))
    return connector.QuantumConnectorWrapper(
        "test",
        str(tmp_path),
        inner,
        provider="dwave",
        providerPool=["dwave", "ibm"],
        providerServiceMap={"dwave": "cpu", "ibm": "cpu2"},
        providerServiceFallbackMap={"dwave": "cpu-backup"},
        providerEnvMap={"dwave": {"EXAMPLE": "value", "MISSING": "${QSPLIT_UNSET_VARIABLE}"}},
        maxConcurrentJobs={"dwave": 1, "ibm": 2},
    )


def location():
    inner = ExecutionLocation(name="local", deployment="local", local=True)
    return ExecutionLocation(name="dwave:local", deployment="test", local=True, service="cpu", wraps=inner)


def test_wrapper_schema_and_configuration(wrapper):
    assert json.loads(wrapper.get_schema())["type"] == "object"
    assert wrapper._provider_env_map == {"dwave": {"EXAMPLE": "value"}}
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
        assert env == {"ORIGINAL": "yes", "QSPLIT_BACKEND": "dwave", "EXAMPLE": "value"}
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
    result = connector.QuantumConnectorWrapper._wrap_iqm_command(command, {"PYTHON_BIN": "/env/python"})
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
