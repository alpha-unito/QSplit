# QSplit

This repository contains a prototype of a **hybrid workflow for combinatorial optimization** formulated as a
**Quadratic Unconstrained Binary Optimization (QUBO)** problem. In QUBO form, the objective is to find a binary vector
$x \in \{0,1\}^n$ that **minimizes a cost function** of the form $x^\top Q x$.

From a quantum computing perspective, the same objective can be interpreted as minimizing the **energy** of an
**Ising/QUBO cost Hamiltonian**: solvers such as quantum annealers or QAOA-like approaches aim to return bitstrings
corresponding to **low-energy configurations** of that Hamiltonian.

## What this repository does

At a high level, the workflow does the following:

1. Reads an input CSV matrix and converts it into a QUBO object, stored as a pickled `QUBO` object in `initial_qubo.pkl`.
2. Uses **QSplit** to decompose the global cost function into several smaller sub-QUBOs (sub-Hamiltonians).
3. Solves each sub-QUBO on a selected backend (e.g., D-Wave, IQM, or compatible solvers) and collects candidate bitstrings.
4. Aggregates partial solutions into a global candidate assignment and reports its **energy**, i.e., the value of the global cost function.

The orchestration is expressed in **CWL** and executed with **Streamflow**, which runs the stages (split, solve, aggregate)
as separate steps.

The Python local runner also supports optional refinement for iterative and quadtree splitting.
See [refinement.md](refinement.md) for configuration, propagation methods, and usage examples.

## Why QUBO splitting is useful today

Many real-world tasks in combinatorial optimization can be written as QUBOs: routing and scheduling, portfolio selection,
facility location, graph partitioning, clustering/feature selection, and more. In principle, these problems can be handed to
quantum or quantum-inspired solvers. In practice:

- current hardware has **limited capacity and non-trivial connectivity**, so a large dense QUBO does not map directly onto a device;
- even classically, solving a single large QUBO can be slow and difficult to scale.

QSplit provides a controlled way to:

- reduce the optimization into sub-problems that fit backend constraints (e.g., variable limits, embedding constraints);
- solve sub-problems on heterogeneous resources (quantum and/or quantum-inspired);
- combine partial results into a global candidate solution and evaluate it via the **global energy/cost**.

The goal is not to claim optimality, but to offer a **reproducible and extensible framework** to study decomposition and hybrid
execution strategies in a realistic setting.

## Software requirements

On the host machine you need:

- Streamflow
- Python 3.12
- Access to the target SLURM partitions (Broadwell/Cascadelake in the provided config)

The runtime no longer depends on Singularity images. Each SLURM template uses
the requested Python virtual environment (`qsplit-cpu` and `qsplit-gpu`) and
fails fast if it is missing.

## Native core and installation with UV

QUBO, splitting, aggregation, conflict resolution, halting heuristics and refinement
run in the C++23 extension `qsplit._core`. Python modules retain the existing imports
and configuration interface. Adapters, datasets, CLI/StreamFlow orchestration and
configuration remain Python. PyMetis still supplies the external METIS partitioner;
Pandas still supplies table conversion, joins, sorting and deduplication.

Supported platforms are **Linux and macOS**. Building from source requires a C++
compiler accepting `-std=c++23` (GCC or Clang), Python development headers and UV.
On macOS install the Xcode Command Line Tools (`xcode-select --install`). On Linux
install your distribution's C++ toolchain and Python development package; on HPC
load the corresponding compiler module before installation. Select a compiler with
`CXX=g++` or `CXX=clang++` if necessary. Windows is not supported.

From the repository root:

```bash
uv sync --python 3.12 --extra dwave
uv run --no-sync python -c "from qsplit.qubo import QUBO; import qsplit._core; print('Native core ready')"
```

UV creates `.venv`, installs the isolated build dependencies (`setuptools` and
`pybind11`), compiles the extension and installs QSplit. No separate library build
or CMake invocation is needed. Choose the extras required by your backends; their
existing incompatibilities still apply. For development and the main local tests:

```bash
uv sync --python 3.12 --extra dev --extra streamflow --extra quantinuum --extra dataset
uv run pytest
```

Python usage is unchanged:

```python
import numpy as np
from qsplit.qubo import QUBO
from qsplit.splitting.split_linear import split_problem

ids = np.arange(3)
qubo = QUBO(np.array([[-1.0, 2.0, 0.0], [0.0, -2.0, 1.0], [0.0, 0.0, -3.0]]), ids, ids)
subproblems = split_problem(qubo, config={"CUT_DIM": 2})
```

`uv run` and `uv sync` track the C++ sources and headers through
`tool.uv.cache-keys`; they rebuild the editable extension when those files change.
After installing the test extras, `uv run pytest` therefore tests the current core.
To force a rebuild explicitly:

```bash
uv pip install --no-deps --reinstall -e .
```

Then restart Python processes that already imported QSplit. Python-only edits need
no recompilation. A missing `_core` import means the extension has not been built
for the active interpreter; install it using that environment's Python.

To distribute a compiled package:

```bash
uv build
uv pip install --python /path/to/environment/bin/python dist/qsplit-*.whl
```

`uv build` creates a source archive and builds its wheel, including the C++ headers.
Build a wheel for each target OS, architecture and Python ABI. A locally built Linux
wheel is not automatically a portable manylinux wheel: build on a compatible target
system and retain its C++ runtime. Compilation does not use `-march=native` or
fast-math. Cluster environments must each have a compatible compiled extension;
a macOS `.so` cannot be copied to Linux.

The binding preserves writable NumPy arrays, DataFrame solutions, dynamic metadata,
`deepcopy` and the dictionary state of existing QUBO pickle files. Matrices are
normalized to float64 and indices to int64 at the boundary; non-contiguous arrays
are supported. Computational loops release the GIL where possible. Do not mutate
shared QUBO arrays while another thread is computing with them.

Equal interaction weights in k-interaction splitting use a stable ordering by
matrix position, instead of NumPy's unspecified default quicksort tie order.
Other voting tie rules, decimal sanitization, padding and energy offsets are
preserved. Floating-point reductions can differ in their final bits from BLAS;
validation compares objectives with numerical tolerances.

Run the reproducible CPU microbenchmarks with
`uv run --no-sync python benchmarks/core.py`. See [TESTING.md](TESTING.md) for
correctness checks and comparisons against a pre-port checkout. These timings cover
core operations, not cloud latency or quantum simulation.

## Run

### 1. Preparing Python environments (optional pre-warm)

Create environments on the cluster login node before running Streamflow:

```bash
python3 -m venv /beegfs/home/fmedina/.venvs/qsplit-cpu
/beegfs/home/fmedina/.venvs/qsplit-cpu/bin/pip install -U pip setuptools wheel
/beegfs/home/fmedina/.venvs/qsplit-cpu/bin/pip install -e "/beegfs/home/fmedina/QSplit[dwave,iqm]"

python3 -m venv /beegfs/home/fmedina/.venvs/qsplit-gpu
/beegfs/home/fmedina/.venvs/qsplit-gpu/bin/pip install -U pip setuptools wheel
/beegfs/home/fmedina/.venvs/qsplit-gpu/bin/pip install -e "/beegfs/home/fmedina/QSplit[ibm-gpu]"
```

### 2. Preparing configuration

QSplit reads runtime settings from explicitly supplied YAML files. The old
configuration environment variables and `providerEnvMap` are no longer supported.
The keys retain their former names (for example `CUT_DIM`, `IQM_TOKEN` and
`REFINEMENT_LOOPS`), but belong in YAML now. Strings, numbers and booleans are
accepted; there is no `${ENV_VAR}` expansion or automatic file discovery.

Copy the public templates before editing:

```bash
mkdir -p configs
cp qsplit.config.template.yaml configs/local.yaml
cp streamflow/cwl/config.template.yml streamflow/cwl/config.yml
cp streamflow/streamflow.template.yml streamflow/streamflow.yml
```

All `.yaml` and `.yml` files are ignored by Git except the three explicitly
allowlisted public templates and existing repository tooling. **Never put real
credentials in a template.** Git ignore prevents ordinary accidental additions;
it cannot prevent `git add -f` or protect files already tracked in another branch.
The previous tracked workflow settings have been replaced with templates.

For local execution, pass a path, an ordered list of paths, or a loaded mapping:

```python
from qsplit.local_runner import qsplit_sampler_refined_iterative

result = qsplit_sampler_refined_iterative(qubo, config="configs/local.yaml")
# Split settings and credentials if needed; later files override earlier keys.
result = qsplit_sampler_refined_iterative(qubo, config=["configs/algorithm.yaml", "configs/solver.yaml"])
```

`QSPLIT_BACKEND` selects the local solver (`dwave` by default, simulated annealing).
`QSPLIT_SOLVER_MODULE` selects a custom module exposing `solve(qubo)`.
Configuration is scoped to the call, including nested splitting/refinement and
adapter calls, and restored after success or failure. For direct component calls,
use `with qsplit.configuration.use("configs/local.yaml"):` or their `config=`
argument where provided. Configuration and credentials are never attached to QUBOs.

Gate-based adapters (IBM QAOA/PCE, IQM, Quantinuum and CUDA-Q) train their
variational parameters exclusively on classical simulators. Set
`LOCAL_TRAINING_QUBITS: 20` in the solver YAML to cap the training circuit size
(default: 20; any positive integer). Larger problems train on an induced
subproblem containing the variables with the largest sums of absolute incident
coefficients, with ties resolved by variable ID. Excluded variables are fixed to
zero during training. The learned beta/gamma angles are shared by layer and reused
on every qubit of the complete final circuit. This transfer is a heuristic and
does not guarantee the same solution quality as training on the full problem.

PCE + QAOA encodes the canonical binary objective (including linear terms) using
three-body X/Y/Z correlations and an auxiliary Ising node. Negative variable IDs
are padding fixed to zero; repeated IDs are accumulated. It uses a QAOA-style
ansatz with three layers, an X mixer and a phase separator consisting of a ZZ
chain plus distinct Z fields. The fields remove the global-X parity and chain
reflection restrictions of a ZZ-only circuit. The actual QUBO objective is in
the nonlinear PCE loss, not the expectation of the phase separator. This is a
heuristic variational solver, with no guarantee of finding the global optimum.

The same six layer angles, encoding and normalized loss are used during local
training, with two reproducible COBYLA starts and up to 200 function evaluations
per start. Positive rescaling of all QUBO coefficients leaves the loss unchanged.
The training limit counts encoded qubits, including capacity for the auxiliary
node, and must be at least 3. Only the final observable evaluation is submitted
to the selected backend; PCE batches the X/Y/Z observables in that evaluation.
Decoding treats means within twice the backend's requested precision (at least
`1e-8`) as uncertain, tries at most four sign assignments, applies one-bit local
search, and selects the candidate with the lowest original QUBO energy. Constant
objectives return immediately without optimization or backend execution.
Simulator adapters also apply the training cap, but still simulate the full
circuit for their final result. `QUANTUM_TUNE_IQM` is obsolete and ignored; IQM
cache entries are separated by the local-training version and qubit limit.

All six CLI entry points accept repeatable `--config FILE` arguments:

```bash
cli_split --config configs/algorithm.yaml --input-matrix input.csv
cli_scatter --config configs/solver.yaml --input-qubo subproblem.pkl --output-qubo solved.pkl
```

Explicit CLI arguments take precedence over their YAML settings (`--cut-dim` and
`--backend`). Without a config, CLI defaults still work for local simulators;
local runner strategies require `CUT_DIM`. Missing hardware credentials fail
explicitly. Use absolute paths for storage directories in distributed runs:
paths *inside* a YAML file refer to the execution host and are not staged by CWL.

### 3. Executing the workflow

Edit `streamflow/cwl/config.yml` to select a JSONL `dataset`, `cut_dim`, enabled
solver lanes, and an absolute `solutions_store_dir`. Edit `streamflow/streamflow.yml`
to configure deployments and Python executables for the target machines.

The workflow accepts separate lists of YAML `File` inputs. CWL stages only the
files routed to each step:

| Input | Recipient steps |
| --- | --- |
| `split_configs` | split |
| `parallel_configs` | parallel solver lane |
| `iqm_configs` | IQM solver lane |
| `quantinuum_h2_configs`, `quantinuum_h2e_configs` | corresponding solver lane |
| `aggregate_configs` | aggregate |
| `storage_configs` | prepare dataset, persist solution, collect results |

For example, create `configs/iqm.yaml` with only the IQM settings and credentials
from the public template, then add this to the CWL settings:

```yaml
iqm_configs:
  - class: File
    path: ../../configs/iqm.yaml
```

Do not route a file containing all providers' credentials to every step. The
`parallel_configs` files are shared by the providers in that lane's pool; use a
dedicated lane/deployment when credentials must be isolated by provider.

The quantum connector's `providerConfigMap` maps providers to a YAML path or list
of paths **used only by the supervisor** for availability probes and cancellation.
Those paths are relative to the StreamFlow configuration directory and are not
copied to workers. Supply worker settings separately through the CWL inputs above.
`providerPythonMap` selects an explicit Python executable per provider, including
an isolated IQM environment, without exporting configuration variables. For IQM,
use matching credentials and a shared `QSPLIT_IQM_STATE_DIR` on supervisor and worker
when supervisor cancellation is required. Pool selection and concurrency remain
in the StreamFlow deployment YAML.

```bash
streamflow run streamflow/streamflow.yml
```

Completed instance solutions are persisted in `solutions_store_dir`; subsequent
runs skip valid existing solutions. StreamFlow also returns the collected results
and manifests as workflow outputs.

## Output

The file `solutions.csv` reports candidate solutions (bitstrings) together with their associated **energy**.
In QUBO/Ising terms, the energy is the value of the **global cost function** for the reported bitstring; equivalently, it is
the energy of the corresponding configuration under the **cost Hamiltonian**.

Columns:

- `node_id`: identifier of the node/sub-problem (e.g., `root`, `root_0`, `root_1`, ...)
- `backend`: backend that produced the solution (e.g., `dwave`, `iqm`, `aggregate`)
- `bitstring`: binary assignment returned by the backend
- `energy`: cost/energy value (lower is better)

Example:

```csv
node_id,backend,bitstring,energy
root,aggregate,10111100010110101110100100100000,-80.27
root_0,iqm,1011110001011010,-188.07
root_1,dwave,0100000000001111,-209.422003363
root_2,iqm,1110110100100000,-154.85
```

Interpretation:

- Each `root_k` line is a backend-produced candidate for a sub-QUBO.
- The `root,aggregate,...` line is the aggregated global candidate assignment.
- Optimization quality is assessed by the **energy**: the workflow is designed to drive the overall solution towards
  lower values of the global cost function / cost Hamiltonian.

## Local tests

The test suite includes unit, integration and end-to-end tests using CPU simulators
and local StreamFlow deployments. It requires no HPC or QPU access.

```bash
uv venv --python 3.12
uv pip install -e ".[dev,streamflow,quantinuum,dataset]"
uv run --no-sync python -m pytest --cov --cov-report=term-missing --cov-report=html
```

See [TESTING.md](TESTING.md) for test levels, isolated IQM/CUDA-Q simulators,
coverage reports and the CI commands.
