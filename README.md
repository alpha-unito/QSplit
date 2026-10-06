# QSplit

QSplit is a framework for solving QUBO optimization problems across classical and
quantum backends. It splits a binary objective into smaller problems, solves them,
aggregates their solutions and optionally refines the global result.

[CWL](streamflow/cwl/instance.cwl) defines the workflow; StreamFlow decides where
its steps run. Numerical algorithms are implemented in C++23 and exposed through
Python. Results are evaluated on the original objective, `x.T @ Q @ x + offset`.
The methods are heuristics and do not guarantee an optimum.

- [Choose a pipeline](workflows.md): compatible split, aggregate and refinement methods.
- [Run tests](TESTING.md): local checks for development.

## Run your first workflow

Use Linux or macOS, UV, Python 3.12 or newer, and a C++23 compiler. On macOS,
install the Xcode Command Line Tools; on Linux, install a C++ toolchain and Python
development headers. Installation builds the native extension automatically.

From the repository root:

```bash
uv sync --locked --python 3.12 --extra dwave --extra streamflow
cp streamflow/streamflow.local.template.yml streamflow/streamflow.local.yml
cp streamflow/cwl/instance.config.template.yml streamflow/cwl/instance.config.yml
mkdir -p streamflow/work
uv run --no-sync streamflow run --debug \
  --outdir reports/local-workflow streamflow/streamflow.local.yml
```

This example uses the bundled CSV, simulated annealing, linear splitting,
belief-propagation aggregation and two conditioned refinement sweeps.
It runs entirely locally and requires no QPU or cluster access.

## Configure your run

There are two configuration layers:

| File | What to change |
| --- | --- |
| `streamflow/cwl/instance.config.yml` | Input CSV, pipeline methods, block sizes, refinement budget and solver settings files |
| `streamflow/streamflow.local.yml` | Deployments, step bindings, provider selection and concurrency |

Start by changing `input_matrix.path` in the instance settings to your square CSV
matrix. See [workflows.md](workflows.md) for valid method combinations.
Set `refinement_method: none` or `refinement_loops: 0` for a single pass.

To run on other machines, change the StreamFlow deployments and bindings.
The [HPC template](streamflow/streamflow.template.yml) provides an example.
Install QSplit and the required backend extras on the workers. Keep credentials
in private solver YAML files, supplied through the corresponding solver config
inputs; use [qsplit.config.template.yaml](qsplit.config.template.yaml) as a reference.
Settings are supplied explicitly through YAML, rather than environment variables.

## Run a dataset

[main.cwl](streamflow/cwl/main.cwl) runs the same configurable pipeline over a JSONL
dataset. Copy [config.template.yml](streamflow/cwl/config.template.yml) to
`streamflow/cwl/config.yml`, then set `dataset` and an absolute
`solutions_store_dir` accessible to the storage steps.

In the StreamFlow configuration, select `cwl/main.cwl` and `cwl/config.yml`.
Prefix instance step bindings with `/qsplit_instances`, as in the HPC template.
Completed instances are skipped on subsequent runs using the same solutions store.
Use a new store when comparing pipeline configurations.

## Read the results

The single-instance workflow exports:

| Output | Contents |
| --- | --- |
| `final_solutions` | CSV with `node_id`, `backend`, `bitstring` and `energy`; the `root,aggregate` rows describe the global result |
| `final_history` | Initial global energy followed by energies of completed refinement rounds |
| `final_state` | Serialized QUBO, best solutions and refinement state |

Lower global energy is better. With refinement enabled, the returned result is the
best encountered, even if a later round worsens. The dataset workflow collects
these results and adds manifests for the completed instances.

Python users can access the same algorithms through `qsplit.splitting`,
`qsplit.aggregation` and `qsplit.refinement`; see the [Python entry points](workflows.md#python-entry-points).
