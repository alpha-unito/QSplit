import asyncio
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import yaml

from qsplit import configuration, local_runner
from qsplit.cwl.cli import scatter
from qsplit.cwl.cli.utils import load_qubo, save_qubo
from streamflow.quantum import qmetrics
from streamflow.quantum.plugin.connector import helpers

REPO = Path(__file__).resolve().parents[2]


def test_layered_yaml_and_scope_restoration(tmp_path):
    common = tmp_path / "common.yaml"
    secret = tmp_path / "secret.yml"
    common.write_text("CUT_DIM: 2\nREFINEMENT_LOOPS: 3\nIQM_TOKEN: first\n")
    secret.write_text("IQM_TOKEN: private-value\nQUANTUM_TUNE_IQM: true\n")
    previous = dict(configuration.current())
    with pytest.raises(RuntimeError):
        with configuration.use([common, secret]):
            assert configuration.require("CUT_DIM") == "2"
            assert configuration.require("IQM_TOKEN") == "private-value"
            assert configuration.get("QUANTUM_TUNE_IQM") == "True"
            with configuration.use({"CUT_DIM": 4}):
                assert configuration.get("IQM_TOKEN") is None
            assert configuration.require("IQM_TOKEN") == "private-value"
            raise RuntimeError("solver failed")
    assert configuration.current() == previous


@pytest.mark.parametrize(
    "text", ["[]", "42", "1: x", "IQM_TOKEN: [secret]", "IQM_TOKEN: [secret", "!!python/object:object {}"]
)
def test_invalid_yaml_does_not_echo_secrets(tmp_path, text):
    config = tmp_path / "private.yaml"
    config.write_text(text)
    with pytest.raises(ValueError) as error:
        configuration.load(config)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("cut", [0, -1, 1.5, True, None, "invalid"])
def test_invalid_cut_dimension_rejected_before_execution(cut):
    with pytest.raises(ValueError, match="CUT_DIM must be a positive integer"):
        configuration.load({"CUT_DIM": cut})


def test_missing_file_is_not_silently_ignored(tmp_path):
    with pytest.raises(FileNotFoundError):
        configuration.load(tmp_path / "missing.yaml")


def test_environment_is_ignored(monkeypatch):
    for key, value in {"CUT_DIM": "999", "IQM_TOKEN": "environment-secret", "QSPLIT_BACKEND": "iqm"}.items():
        monkeypatch.setenv(key, value)
    with configuration.use({}):
        assert scatter.resolve_backend("") == "dwave"
        with pytest.raises(ValueError, match="CUT_DIM"):
            configuration.require("CUT_DIM")
        with pytest.raises(RuntimeError, match="IQM auth"):
            qmetrics._resolve_iqm_auth()


def test_concurrent_configurations_do_not_leak():
    async def scenario():
        async def task(value):
            with configuration.use({"IQM_TOKEN": value}):
                await asyncio.sleep(0)
                return configuration.require("IQM_TOKEN")

        assert await asyncio.gather(task("first"), task("second")) == ["first", "second"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "sampler",
    [
        local_runner.qsplit_sampler_recursive,
        local_runner.qsplit_sampler_iterative,
        local_runner.qsplit_sampler_interactions,
        local_runner.qsplit_sampler_graph_partitioning,
        local_runner.qsplit_sampler_quadtree,
        local_runner.qsplit_sampler_refined_iterative,
        local_runner.qsplit_sampler_refined_quadtree,
    ],
)
def test_local_runner_accepts_yaml_without_serializing_secrets(tmp_path, make_qubo, sampler, assert_solution):
    config = tmp_path / "local.yaml"
    config.write_text("CUT_DIM: 2\nQSPLIT_SOLVER_MODULE: qsplit.adapters.all_zero\nIQM_TOKEN: secret-sentinel\n")
    qubo = sampler(make_qubo([[-1, -1, -1], [0, -1, -1], [0, 0, -1]]), config=config)
    assert_solution(qubo)
    assert (qubo.solutions.energy == 0).all()
    output = tmp_path / "result.pkl"
    save_qubo(output, qubo)
    assert b"secret-sentinel" not in output.read_bytes()
    assert configuration.get("IQM_TOKEN") is None


def test_cli_layering_and_argument_precedence(tmp_path, monkeypatch, make_qubo):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("QSPLIT_BACKEND: iqm\nQSPLIT_SOLVER_MODULE: invalid.module\n")
    second.write_text("QSPLIT_SOLVER_MODULE: qsplit.adapters.all_zero\nIQM_TOKEN: secret-sentinel\n")
    source, result = tmp_path / "source.pkl", tmp_path / "result.pkl"
    save_qubo(source, make_qubo([[-1]]))
    argv = [
        "cli_scatter",
        "--config",
        str(first),
        "--config",
        str(second),
        "--backend",
        "dwave",
        "--input-qubo",
        str(source),
        "--output-qubo",
        str(result),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    scatter.main()
    assert load_qubo(result).backend == "dwave"
    assert load_qubo(result).solutions.energy.tolist() == [0]
    assert b"secret-sentinel" not in result.read_bytes()
    assert sys.argv == argv
    assert configuration.get("IQM_TOKEN") is None


def test_iqm_metrics_passes_yaml_token_explicitly(monkeypatch):
    iqm = ModuleType("iqm")
    client = ModuleType("iqm.iqm_client")
    client.IQMClient = Mock()
    monkeypatch.setitem(sys.modules, "iqm", iqm)
    monkeypatch.setitem(sys.modules, "iqm.iqm_client", client)
    with configuration.use(
        {"IQM_TOKEN": "yaml-token", "IQM_SERVER_URL": "https://example.invalid", "IQM_QUANTUM_COMPUTER": "garnet"}
    ):
        qmetrics.get_iqm_quantum_backend()
    client.IQMClient.assert_called_once_with("https://example.invalid", token="yaml-token", quantum_computer="garnet")


def test_iqm_probe_uses_only_selected_provider_configuration(monkeypatch):
    def probe():
        assert configuration.require("IQM_TOKEN") == "iqm-secret"
        assert configuration.get("TOKEN_IBM") is None
        return object()

    monkeypatch.setattr(qmetrics, "get_iqm_quantum_backend", probe)
    monkeypatch.setattr(qmetrics, "get_quantum_metrics", lambda *args: {"active": True, "queue": 3})
    assert helpers.fetchProviderStateFor(
        "iqm", ["iqm"], lambda _: True, {"iqm": {"IQM_TOKEN": "iqm-secret"}, "ibm": {"TOKEN_IBM": "ibm-secret"}}
    ) == (True, 3)
    assert configuration.get("IQM_TOKEN") is None


def test_subprocess_probe_keeps_secrets_out_of_argv_and_environment(monkeypatch):
    run = Mock(return_value=type("Result", (), {"returncode": 0, "stdout": '{"active": true}'})())
    monkeypatch.setattr(helpers.subprocess, "run", run)
    assert helpers._probe_iqm_metrics_via_subprocess({"iqm": {"IQM_TOKEN": "secret-sentinel"}}, "/venv/bin/python") == {
        "active": True
    }
    args, kwargs = run.call_args
    assert "secret-sentinel" not in repr(args)
    assert "env" not in kwargs
    assert "secret-sentinel" in kwargs["input"]


def test_workflow_routes_files_only_to_requested_steps():
    workflow = yaml.safe_load((REPO / "streamflow/cwl/instance.cwl").read_text())
    expected = {
        "split": "split_configs",
        "parallelize": "parallel_configs",
        "iqm": "iqm_configs",
        "quantinuum_h2": "quantinuum_h2_configs",
        "quantinuum_h2e": "quantinuum_h2e_configs",
        "aggregate": "aggregate_configs",
        "persist_solution": "storage_configs",
    }
    for step, field in expected.items():
        inputs = workflow["steps"][step]["in"]
        assert inputs["configs"] == field
        assert [v for v in inputs.values() if isinstance(v, str) and v.endswith("_configs")] == [field]
        assert workflow["inputs"][field] == {"type": "File[]", "default": []}
    assert "configs" not in workflow["steps"]["merge_solved"]["in"]


def test_private_yaml_is_ignored_anywhere_and_templates_are_not():
    private = [
        "config.yaml",
        "private.YAML",
        "private.YmL",
        "secrets.yml",
        "configs/iqm.yaml",
        "nested/arbitrary.yml",
        "streamflow/streamflow.yml",
        "streamflow/cwl/config.yml",
        "nested/config.template.yaml",
    ]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        input="\n".join(private),
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.splitlines() == private
    tracked = subprocess.run(
        ["git", "ls-files", "--", "streamflow/streamflow.yml", "streamflow/cwl/config.yml"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )
    assert not tracked.stdout, "Operational YAML must be removed from the Git index, not only ignored"
    public = [
        "qsplit.config.template.yaml",
        "streamflow/streamflow.template.yml",
        "streamflow/cwl/config.template.yml",
        ".github/workflows/ci.yml",
        ".pre-commit-config.yaml",
    ]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        input="\n".join(public),
        cwd=REPO,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1 and not result.stdout
    configuration.load(REPO / public[0])


def test_integral_yaml_float_is_normalized_before_consumers_parse_it():
    with configuration.use({"CUT_DIM": 2.0}):
        assert int(configuration.require("CUT_DIM")) == 2


@pytest.mark.parametrize("tune", [False, True])
def test_iqm_adapter_passes_yaml_credentials_and_tuning_to_sdk(tmp_path, monkeypatch, make_qubo, tune):
    import runpy

    provider_module = ModuleType("iqm.qiskit_iqm")
    provider_module.IQMProvider = Mock()
    fake_module = ModuleType("iqm.qiskit_iqm.fake_backends.fake_garnet")
    fake_module.IQMFakeGarnet = Mock()
    util_module = ModuleType("qsplit.adapters.iqm.util")
    util_module.get_qaoa_circuit_optimized = Mock(return_value=("circuit", {}, []))
    util_module.run_quantum_optimizer = Mock(return_value={"0": 10})
    util_module.to_dataframe = Mock(return_value="result")
    for name, module in [
        ("iqm", ModuleType("iqm")),
        ("iqm.qiskit_iqm", provider_module),
        ("iqm.qiskit_iqm.fake_backends.fake_garnet", fake_module),
        ("qsplit.adapters.iqm.util", util_module),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    solve = runpy.run_path(str(REPO / "qsplit/adapters/iqm/iqm_qaoa_q.py"))["solve"]
    config = tmp_path / "iqm.yaml"
    config.write_text(yaml.safe_dump({"IQM_TOKEN": "yaml-token", "QUANTUM_TUNE_IQM": tune}))
    assert solve(make_qubo([[-1]]), config=config) == "result"
    provider_module.IQMProvider.assert_called_once_with(
        url="https://resonance.meetiqm.com/", token="yaml-token", quantum_computer="garnet"
    )
    optimizer_backend = util_module.get_qaoa_circuit_optimized.call_args.kwargs["backend"]
    assert optimizer_backend is (
        provider_module.IQMProvider.return_value.get_backend.return_value
        if tune
        else fake_module.IQMFakeGarnet.return_value
    )


def test_cache_only_never_submits_when_cache_is_disabled(tmp_path, monkeypatch, make_qubo):
    config, source, output = tmp_path / "iqm.yaml", tmp_path / "source.pkl", tmp_path / "output.pkl"
    config.write_text("QSPLIT_BACKEND: iqm\nQSPLIT_IQM_SUBPROBLEM_CACHE: false\n")
    save_qubo(source, make_qubo([[-1]]))
    monkeypatch.setattr(scatter, "load_solver", Mock(side_effect=AssertionError("Must not submit a job")))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli_scatter",
            "--config",
            str(config),
            "--cache-only",
            "--input-qubo",
            str(source),
            "--output-qubo",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as error:
        scatter.main()
    assert error.value.code == 75
    assert not output.exists()
