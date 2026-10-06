# Local testing

The default suite never submits cluster jobs or uses real QPUs. It runs small
problems with CPU simulators and uses fake responses for external scheduling,
provider failures and job cancellation. Dataset generator tests supply local
fixture data instead of downloading a dataset.

## Install and run

CI exercises Python 3.12, 3.13 and 3.14. For example, to use Python 3.12:

```bash
uv sync --locked --python 3.12 --extra dev --extra streamflow --extra quantinuum --extra dataset
uv run pytest
```

`--no-sync` preserves the extras installed in the environment. No provider tokens,
SLURM installation, GPU, containers or HPC access are needed. Tests use temporary
directories, isolate YAML configuration, clear inherited provider credentials and
block non-local socket connections in the pytest process. Subprocess tests use
explicit local deployments and CPU backend selections. Tests do not modify the
production StreamFlow configuration or the repository's solution caches.

Optional SDK tests are skipped when their dependencies are absent. The CI installs
the required extras, including separate jobs for IQM and CUDA-Q, so those paths
are exercised rather than silently omitted.

The C++23 extension is compiled by the installation command. See the native build
requirements in [README.md](README.md). After installing the test extras,
`uv run pytest` automatically rebuilds when C++ sources or headers change. To force
a rebuild, use `uv pip install --no-deps --reinstall -e .`. Linux and macOS are supported;
Windows jobs have been removed.

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

## PCE + QAOA regressions

Run the exact local regressions independently of backend smoke tests:

```bash
uv run --no-sync python -m pytest tests/unit/test_pce_symmetry.py -q -s
```

The original PCE circuit combined `k=3`, a uniform ZZ chain, the default X mixer
and `|+>^q`. It preserved global-X parity, forcing every three-body all-Y and
all-Z mean to zero; at three qubits XXX was fixed at +1 and the loss was constant.
The corrected phase separator adds distinct longitudinal Z fields
`0.5 + (i + 1)/(q + 1)`. This breaks both global-X parity and chain reflection
while retaining three QAOA layers and six shared angles for subproblem transfer.
The driver is problem-independent: the QUBO is optimized through the nonlinear
PCE loss, not through the driver's energy expectation.

The suite now checks:

- Every encoded observable, including the auxiliary spin, varies with the
  parameters on 1, 2, 5, 11 and 29 binary variables. The former expected failure
  is now a normal regression test. Mirror-related X observables can differ.
- The three-qubit loss is nonconstant, diagonal coefficients affect it, and
  positive rescaling preserves it. Couplings are normalized by their largest
  absolute value; the regularizer uses the sum of normalized absolute weights.
  This regularization is a heuristic, not an optimality bound for signed QUBOs.
- Exhaustive enumeration preserves the QUBO-to-auxiliary-Ising energy identity
  with offsets, both auxiliary signs, asymmetric coefficients, disjoint or
  repeated variable IDs and negative padding. Output energies are recomputed
  against the original objective.
- Tiny sign perturbations do not arbitrarily change decoding. Uncertain spins
  use two common fillings, and an uncertain auxiliary spin uses both orientations,
  yielding at most four candidates. Each receives the same one-bit local search;
  the lowest original QUBO energy wins. The tolerance uses twice the backend's
  requested precision, with a roundoff floor of `1e-8`. This does not enumerate
  every combination of ambiguous bits or guarantee the best possible decoding.
- One-bit polishing lowers energy, ends at a local minimum, and handles small
  coefficient scales. Constant objectives bypass the optimizer and backend.
- Local training uses the canonical objective, honors the encoded-qubit cap and
  transfers the six angles to one final full-problem evaluation. A constant
  induced training problem uses the initial angles without optimizing a
  regularizer-only objective.

To check actual simulator execution and tiny instances with known optima:

```bash
uv run --no-sync python -m pytest tests/integration/test_simulators.py -k pce -q
```

Small exact-optimum regressions verify selected cases, not a general guarantee.
For broader quality experiments, use several instances and seeds and report
`E - E_opt` separately for raw sign decoding and after one-bit search. Compare
against all-zero and random starts with the same polishing, under the same
training evaluation budget. Measure capped-subproblem transfer separately from
full-problem training; a passing symmetry regression does not establish quantum
advantage or good approximation ratios on larger instances.

Qiskit's [QAOA API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/2.3/qiskit.circuit.library.qaoa_ansatz)
documents the default state and mixer. The
[IBM PCE tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/pauli-correlation-encoding-for-qaoa)
uses an RY/RZ `efficient_su2` ansatz; the suite retains this as an independent
control with the same observables.

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

CUDA-Q tests use Linux and Python 3.12. The `cudaq` extra installs NVIDIA's
`cuda-quantum-cu12` binary distribution directly, so the lockfile includes the
actual `cudaq` module and its dependencies. The `cudaq` metapackage chooses its
binary dependency dynamically and previously produced an incomplete lockfile.
This extra targets CUDA 12; the CPU test does not require a GPU.

```bash
UV_PROJECT_ENVIRONMENT=.venv-cudaq uv sync --locked --python 3.12 --extra dev --extra cudaq
.venv-cudaq/bin/python -c "import cudaq"
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

The configuration includes both **line and branch coverage** for the Python code
in `qsplit` and the repository's `streamflow.quantum` plugin. `pytest-cov` does not
measure C++ coverage; its percentage must not be interpreted as native core coverage. Hardware adapters remain in the
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
also writes JUnit XML. Each `local-test-reports-<os>-py<version>` artifact contains these results,
unit-only and cumulative JSON reports, final XML, HTML and the coverage database.
It also records the platform, architecture, Python and installed dependency
versions, so differences between runners can be investigated.
The 75% gate applies to the combined line/branch coverage of the main local suite,
not to each individual level. Separate IQM/CUDA-Q jobs publish their own coverage
and JUnit reports; their percentages describe those isolated runs and are not
added arithmetically to the main report.

The main matrix has six required jobs: Linux and macOS, each with
Python 3.12, 3.13 and 3.14. Each job runs unit, integration and E2E tests and
produces its own cumulative coverage report with the 75% gate. `fail-fast: false`
lets other combinations finish when one fails; no combination is allowed to
fail silently.

IQM and CUDA-Q retain their isolated Linux/Python 3.12 jobs. In particular, the
IQM extra currently pins its SDK to Python < 3.13; the six-job main matrix does
not claim coverage of these two optional SDKs.

On Linux, the end-to-end step uses at most two CPUs (`taskset`) to exercise
resource contention. macOS does not invoke this Linux-specific tool.
The local CWL test configures the scheduler's `retry_delay` to one
second: local deployments share CPU capacity, but StreamFlow's default scheduler
only notifies waiting jobs on the deployment that released resources. Without a
periodic recheck, another deployment can wait indefinitely on a small runner.
This retries resource allocation, not failed solver commands.

The command timeout remains 180 seconds. E2E commands write combined stdout/stderr
logs in their pytest temporary directory, and StreamFlow runs with `--debug`.
Failures include the log tail; timeouts also terminate the subprocess group on
Linux/macOS. CI uses `--basetemp=reports/e2e-work`, so complete command logs and
test inputs are included in the existing report artifact even after failure.
That directory is cleared by pytest at the start of each invocation; use a
dedicated directory when reproducing this option locally.

Partitioning and stochastic samplers may produce different valid assignments
across architectures or native library builds. Tests check mathematical
invariants instead of identical partitions. Regression tests explicitly exercise
zero-matrix graph partitions: their dummy solutions contain `NaN` (no vote),
which aggregators must ignore rather than convert to an integer or count as a
vote for zero.

CI installs dependencies from `uv.lock` with `uv sync --locked`, including Ruff.
The pre-commit Ruff revision matches the locked version; update it together with
any Ruff lockfile upgrade. `uv pip install -e ".[dev]"` resolves dependencies anew
and can select a newer formatter. Run the same lint and format checks locally with:

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

The configuration regression tests cover YAML validation and layering, scoped local
samplers, ignored legacy environment values, explicit provider authentication,
secret-free QUBO serialization, Git ignore rules and per-step CWL routing. The
StreamFlow E2E runs both with defaults and with two staged solver YAML files, then
checks persisted results and resume in both modes.

## Native core checks and benchmarks

The original algorithm tests run against the compiled implementation. Additional
`tests/unit/test_native_core.py` cases check strided arrays, invalid mutable shapes,
legacy pickle states, metadata/deepcopy, exact conflict resolution against exhaustive
enumeration, solver callbacks and fractional conditioning against the full expected
objective. Native failures propagate as Python exceptions. CI also runs `uv build`
to verify that the source distribution includes everything needed for compilation,
then installs and imports the wheel outside the source tree.

```bash
uv run --no-sync python -m pytest tests/unit/test_native_core.py tests/unit/test_algorithms.py
uv run --no-sync python benchmarks/core.py --repeat 5
```

For a comparison, use the same interpreter, dependencies and machine, pointing the
benchmark at a separate checkout containing the Python implementation:

```bash
uv run --no-sync python benchmarks/core.py --source-root /path/to/pre-port-checkout --repeat 5
```

Inputs use a fixed seed, include NumPy/Pandas boundary conversion, and report median
wall-clock milliseconds after a warm-up. There are no timing assertions in CI.
The benchmark does not exercise providers, the network or large quantum simulators.

Measured locally on 2026-09-29, macOS 27 arm64, Python 3.12.7, NumPy 2.3.4,
Pandas 2.3.3 and Clang 23.1.0, against the Python core from `e14ddf8`. These are
medians of five runs after warm-up, without concurrent builds or tests:

| Operation | Python (ms) | C++ binding (ms) | Speedup |
| --- | ---: | ---: | ---: |
| QUBO construction (128 variables) | 0.165 | 0.038 | 4.3× |
| Conditioning (128 variables, window 16) | 0.366 | 0.091 | 4.0× |
| Active variable count (128 variables) | 0.270 | 0.049 | 5.5× |
| Linear aggregation (8 × 128 samples × 128 variables) | 539.012 | 6.911 | 78.0× |
| Closest assignments (64 × 128 candidates, 32 variables) | 277.307 | 0.444 | 624.6× |
| Exact conflicts (8 samples, 10 missing variables) | 72.376 | 1.288 | 56.2× |
| Quadtree splitting (128 variables, cut 16) | 72.246 | 5.925 | 12.2× |

These component timings do not predict total workflow speedup when backend
simulation, cloud latency or filesystem operations dominate.
