# Local testing

The default suite never submits cluster jobs or uses real QPUs. It runs small
problems with CPU simulators and uses fake responses for external scheduling,
provider failures and job cancellation. Dataset generator tests supply local
fixture data instead of downloading a dataset.

## Install and run

Use Python 3.12, the version exercised in CI:

```bash
uv venv --python 3.12
uv pip install -e ".[dev,streamflow,quantinuum,dataset]"
uv run --no-sync python -m pytest
```

`--no-sync` preserves the extras installed in the environment. No provider tokens,
SLURM installation, GPU, containers or HPC access are needed. Tests use temporary
directories, clear provider selection/credentials from their environment and
block non-local socket connections in the pytest process. Subprocess tests use
explicit local deployments and CPU backend selections. Tests do not modify the
production StreamFlow configuration or the repository's solution caches.

Optional SDK tests are skipped when their dependencies are absent. The CI installs
the required extras, including separate jobs for IQM and CUDA-Q, so those paths
are exercised rather than silently omitted.

## Test levels

| Level | Command | Scope |
| --- | --- | --- |
| Unit | `uv run --no-sync python -m pytest -m unit` | QUBO energy invariants, splitting, aggregation, conflict resolution, refinement, CLI validation, allocation, serialization, cache invalidation, metrics, scheduling, retry and job lifecycle |
| Integration | `uv run --no-sync python -m pytest -m integration` | All local runner strategies, real simulated annealing, IBM QAOA/PCE and TKET/Aer, refinement, CLI file pipelines, dataset generation, persistence and resume |
| End-to-end | `uv run --no-sync python -m pytest -m e2e` | Installed CLI commands in subprocesses and the actual dataset CWL workflow executed by StreamFlow with the QSplit connector wrapping a local deployment; a second run verifies resume |

All tests, including the original suite, are grouped in `tests/unit`,
`tests/integration` and `tests/e2e`. The original IBM and D-Wave simulator tests
live with the other integration tests. `tests/conftest.py` applies level markers
based solely on the directory.

Correctness checks recompute `x.T @ Q @ x + offset` independently of adapter
reported energies. Tiny exact enumeration checks mathematical invariants and
provides lower bounds. Stochastic solvers are checked for valid assignments and
correct energies, without requiring identical samples or a global optimum.

## Isolated simulators

IQM has dependency conflicts with StreamFlow and Quantinuum. Install it separately:

```bash
uv venv .venv-iqm --python 3.12
uv pip install --python .venv-iqm/bin/python -e ".[dev,iqm]"
.venv-iqm/bin/python -m pytest tests/integration -m iqm
```

This exercises circuit construction, IQM transpilation, optimization and sampling
using `IQMFakeAdonis`, backed by a local simulator. It does not instantiate a
remote IQM provider.

CUDA-Q tests require a platform with CUDA-Q wheels (the CI uses Linux):

```bash
uv venv .venv-cudaq --python 3.12
uv pip install --python .venv-cudaq/bin/python -e ".[dev,cudaq]"
.venv-cudaq/bin/python -m pytest tests/integration -m cudaq
```

The test substitutes the adapter's GPU target list with `qpp-cpu`, while executing
its real circuit, optimizer, sampler and result conversion. It does not validate
CUDA/GPU execution. Real QPU access, GPU drivers, remote SLURM deployment and live
provider availability remain outside this local suite; retries and provider
control are checked with fake jobs/responses.

## Coverage and CI reports

Integration and end-to-end tests contribute to code coverage exactly as unit
tests do. Repeatedly executing a line does not raise coverage: higher-level tests
increase it when they exercise previously untested paths. Coverage is useful for
finding gaps, but does not measure optimization quality or prove correctness.

The configuration includes both **line and branch coverage**, for all of `qsplit`
and the repository's `streamflow.quantum` plugin. Hardware adapters remain in the
report even when their real hardware cannot be exercised. The subprocess patch
requires `coverage >= 7.10.6` and records the CLI processes launched by CWL.

Generate a cumulative local report:

```bash
uv run --no-sync python -m pytest --cov --cov-report=term-missing \
  --cov-report=html --cov-report=xml:reports/coverage.xml
```

Open `htmlcov/index.html` to inspect uncovered lines and branches. To reproduce the
main CI's separate levels and cumulative coverage:

```bash
uv run --no-sync python -m pytest -m "unit and not iqm and not cudaq" \
  --cov --cov-report=json:reports/coverage-unit.json
uv run --no-sync python -m pytest -m "integration and not iqm and not cudaq" \
  --cov --cov-append --cov-report=json:reports/coverage-unit-integration.json
uv run --no-sync python -m pytest -m e2e \
  --cov --cov-append --cov-fail-under=75 --cov-report=html \
  --cov-report=xml:reports/coverage.xml --cov-report=json:reports/coverage.json
```

The first command resets coverage, the next two append to it. In CI each level
also writes JUnit XML. The `local-test-reports` artifact contains these results,
unit-only and cumulative JSON reports, final XML, HTML and the coverage database.
The 75% gate applies to the combined line/branch coverage of the main local suite,
not to each individual level. Separate IQM/CUDA-Q jobs publish their own coverage
and JUnit reports; their percentages describe those isolated runs and are not
added arithmetically to the main report.

CI also runs Ruff lint and format checks. Run them locally with:

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```
