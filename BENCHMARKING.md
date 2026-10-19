# QOBLIB benchmarks

The default [catalog request](streamflow/cwl/benchmark.all.request.template.yaml) discovers
all classes of a QOBLIB checkout with `instances: all`, then keeps compatible
instances within the configured QUBO budgets. The existing
[benchmark workflow](streamflow/cwl/benchmark.cwl) expands the catalog, converts
instances to QUBO, runs QSplit, and exports native QOBLIB solutions and comparisons
in one `streamflow run`.

To select all instances from specific classes, use their registered names:

```yaml
instances: [labs, independentset]
```

Catalog selection requires the QOBLIB checkout (`--repository` for the local
runner, `qoblib_repository` in CWL). To select individual instances instead, use
the explicit entry format:

```yaml
instances:
  - problem: labs
    instance: '8'
  - problem: independentset
    instance: farm
```

The selection uses native formulations for Market Split, LABS, Birkhoff and
Independent Set; the default LP formulation for the other classes; and every
available binary quadratic portfolio variant, with its budget and risk weight.
Birkhoff collections expand into individual instances. LABS and Network names
are also discovered from models and reference solutions, since they do not have
individual instance files. Instances without a supported LP remain in the catalog.
The checkout commit is recorded in the benchmark provenance.

The default request uses `max_variables: 100000` and
`max_matrix_bytes: 80000000000`, with one run per instance. This is an HPC
ceiling of 80 GB for **one** float64 dense matrix, not a total job-memory estimate.
Set `max_matrix_bytes` to the budget available to a single process on your node;
other matrices, model metadata, coefficients and solver work require extra RAM.
For comparison, a cap of 1,000,000 variables would require 8 TB for one matrix.

Selection checks the full QUBO size before expanding coefficients, including
integer bits, LABS products and inequality slack. Birkhoff's native formulation
enumerates all `n!` permutations and has `n! * (1 + 2 * scale.bit_length())` QUBO
variables; oversized subinstances are rejected before enumerating permutations.
The filter measures compatibility and size, rather than scientific interest or
difficulty. No class is excluded by name. Default StreamFlow settings use the
catalog request; the [individual request](streamflow/cwl/benchmark.request.template.yaml)
remains a small example for testing one instance.

The current bridge cannot solve every instance:
some lack a compatible LP, have continuous/unbounded variables, or exceed the
conversion limits. The complete request uses `on_conversion_error: skip` to run
the convertible instances and record every failure in `conversion_errors` in
`results.json` and the bundle's `manifest.json`. Set it to `error` to fail at the
first conversion error instead. `catalog_count` includes Birkhoff subinstances
and portfolio variants. Increasing a size limit does not add support for another
model type. An empty convertible dataset always fails. Catalog requests also
default to skipping conversion errors when the policy is omitted; explicit
instance lists retain strict error handling unless configured otherwise.

## Install

Run these commands from the QSplit repository root:

```bash
git clone https://github.com/ZIB-AOPT/QOBLIB.git ../QOBLIB
uv sync --locked --extra benchmark --extra dwave --extra streamflow
```

Use an existing complete checkout instead of cloning again. For reproducibility,
check out the desired QOBLIB commit before launching. Cargo on the launching
machine enables QOBLIB's official checkers. With `check: auto`, unavailable
checkers are reported as `UNAVAILABLE`; Sports currently has no official checker
in the Python client. Use `check: required` when unavailable checkers should stop
the export.

## Local Python runner

Copy the pipeline parameters and edit `CUT_DIM`, `SPLIT_METHOD`,
`AGGREGATE_METHOD` and the refinement settings in that YAML before launching:

```bash
mkdir -p configs
cp qsplit.config.template.yaml configs/benchmark.yaml
uv run --no-sync local_benchmark \
  --request streamflow/cwl/benchmark.all.request.template.yaml \
  --repository ../QOBLIB \
  --config configs/benchmark.yaml \
  --output-dir reports/qoblib-local
```

Pipeline parameters are supplied only through YAML; the CLI takes paths to the
request, repository, parameter files and output directory. The local runner uses
`CUT_DIM: 16` by default and the existing D-Wave simulated annealing adapter.
The local runner defaults to zero refinement rounds; pass a YAML with
`REFINEMENT_LOOPS: 2` and `REFINEMENT_METHOD: consensus` through `--config` to
match the workflow example. Results are in `reports/qoblib-local/results/results.json`.
Use a fresh output directory when changing the pipeline.

## Local StreamFlow workflow

```bash
cp streamflow/cwl/benchmark.all.config.template.yml streamflow/cwl/benchmark.config.yml
cp streamflow/streamflow.benchmark.template.yml streamflow/streamflow.benchmark.yml
mkdir -p streamflow/work
```

Edit `streamflow/cwl/benchmark.config.yml` before launching:

```yaml
qoblib_repository:
  class: Directory
  path: /absolute/path/to/QOBLIB
solutions_store_dir: /absolute/path/to/benchmark-all/solutions
cut_dim: 32
```

Keep the other settings from the template, then run:

```bash
uv run --no-sync streamflow run \
  --outdir reports/qoblib-workflow-local streamflow/streamflow.benchmark.yml
```

## Remote StreamFlow workflow over SSH

Install QSplit on the worker at the same revision as the launching machine. On
the worker, from its QSplit checkout:

```bash
uv sync --locked --extra dwave
```

On the launching machine, copy the configured local settings and the remote
deployment example:

```bash
cp streamflow/streamflow.benchmark.remote.template.yml streamflow/streamflow.benchmark.remote.yml
cp streamflow/cwl/benchmark.config.yml streamflow/cwl/benchmark.remote.config.yml
mkdir -p streamflow/work
```

Set `hostname`, `username`, and the absolute remote `workdir` in
`streamflow/streamflow.benchmark.remote.yml`. Set the worker's absolute
`.venv/bin` path in `streamflow/templates/benchmark_ssh.sh`. The SSH service
template activates that environment for both classical and solver steps. SSH
authentication uses the worker connector's normal key/agent configuration.
In `streamflow/cwl/benchmark.remote.config.yml`, set a new `solutions_store_dir`
for the remote run and the desired `cut_dim`; otherwise the existing local
solutions will be resumed rather than recomputed remotely.

```bash
uv run --no-sync streamflow run \
  --outdir reports/qoblib-workflow-remote streamflow/streamflow.benchmark.remote.yml
```

Preparation, persistence, collection and native export run on the launching
machine; split, solve, aggregate and refinement run on the SSH worker. StreamFlow
transfers the step input/output files, so a shared filesystem is not required.
`qoblib_repository.path` and `solutions_store_dir` refer to the launching machine.
The backend is simulated annealing, also in this remote example. For an HPC
queue, replace the SSH deployment with the SSH/Slurm arrangement in
[the HPC template](streamflow/streamflow.template.yml), retaining the benchmark
step prefix `/qsplit/qsplit_instances/`.

## Set the block size

| Execution | Setting |
| --- | --- |
| `local_benchmark` | `CUT_DIM: 32` in the YAML passed through `--config` |
| StreamFlow local | `cut_dim: 32` in `streamflow/cwl/benchmark.config.yml` |
| StreamFlow remote | `cut_dim: 32` in `streamflow/cwl/benchmark.remote.config.yml` |
| Refinement with a different block size | `refinement_cut_dim` in the CWL settings; otherwise it inherits `cut_dim` |

The request YAML chooses instances, repetitions, checker policy and conversion
limits. `cut_dim` belongs to the pipeline settings, not the request.
`max_variables` caps the full QUBO dimension, while `max_matrix_bytes` caps the
memory for its dense matrix (approximately `8 * dim**2` bytes per matrix).
Splitting occurs after conversion, so lowering `cut_dim` does not reduce those
conversion requirements. Use a new solutions store and output directory for a
new comparison; the workflow resumes completed instance IDs from its store.

StreamFlow exports a `report` file (`results.json`) plus native solutions and
submission summary CSVs in `benchmark_results`. Compare original objectives,
not QUBO energies containing penalties. Review the reported conversion failures
and official checker verdicts before treating the run as a complete benchmark
or preparing a submission.

## Progress output

The benchmark prints flushed status messages to stderr while discovering the
catalog, converting instances, solving runs and exporting native solutions.
Messages include the class and instance, one-based run number, position in the
current phase, remaining items, conversion failures and elapsed solve time.
For example:

```text
QSPLIT BENCHMARK prepare | 17/100 remaining=83 | marketsplit/ms_03_050_002 | loading instance
QSPLIT BENCHMARK solve | 3/50 remaining=47 | labs/labs008 | run=3 | starting dim=36
QSPLIT BENCHMARK solve | labs/labs008 | run=3 | split completed; solving 4 subproblems
```

`remaining` counts items after the current item, within that phase. The local
runner displays messages immediately. StreamFlow captures command output and
prints it when each step finishes; use `streamflow run --debug` for its command
and step diagnostics as well.
