# Local checks

Tests use CPU simulators, temporary files and local StreamFlow deployments.
They require no QPU, cluster access or provider credentials.

## Install and run

```bash
uv sync --locked --python 3.12 \
  --extra dev --extra streamflow --extra quantinuum --extra dataset
uv run --no-sync python -m pytest
```

Installation builds the C++ extension. Use `--no-sync` to keep the installed
extras; use `uv run` without that flag to rebuild after native source changes.
Optional SDK tests are skipped when their dependencies are absent.

| Scope | Command |
| --- | --- |
| Unit checks | `uv run --no-sync python -m pytest -m unit` |
| Integration checks | `uv run --no-sync python -m pytest -m integration` |
| CLI and actual local CWL workflows | `uv run --no-sync python -m pytest -m e2e` |
| Formatting and lint | `uv run --no-sync ruff check .` and `uv run --no-sync ruff format --check .` |

For pipeline changes, focus on
[distributed refinement tests](tests/integration/test_distributed_refinement.py) and
[StreamFlow E2E tests](tests/e2e/test_workflows.py). They cover method combinations,
file transfers, sequential conditioned updates, state/history outputs and resume.
The [native core tests](tests/unit/test_native_core.py) check C++ bindings,
serialization and numerical invariants. Tests verify global energies and valid
assignments; passing them does not establish optimization quality or quantum advantage.

## Optional simulator environments

IQM conflicts with the main StreamFlow/Quantinuum environment. Test it separately:

```bash
uv venv .venv-iqm --python 3.12
uv pip install --python .venv-iqm/bin/python -e ".[dev,iqm]"
.venv-iqm/bin/python -m pytest tests/integration -m iqm
```

CUDA-Q simulator tests use Linux and Python 3.12:

```bash
UV_PROJECT_ENVIRONMENT=.venv-cudaq uv sync --locked --python 3.12 --extra dev --extra cudaq
.venv-cudaq/bin/python -m pytest tests/integration -m cudaq
```

## Coverage and CI

```bash
uv run --no-sync python -m pytest --cov --cov-report=term-missing \
  --cov-report=html --cov-report=xml:reports/coverage.xml
```

Open `htmlcov/index.html` for Python coverage; this report does not measure C++
coverage. CI checks Linux/macOS with Python 3.12–3.14 and runs the optional simulators
in separate jobs. See [ci.yml](.github/workflows/ci.yml) for the current matrix,
coverage gate, build checks and report artifacts.
