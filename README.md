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


### 2. Preparing the input QUBO

The repository ships an example matrix declared in the `cwl/config.yml` that will be QUBO serialised in `initial_qubo.pkl` during the first step. To experiment with different inputs, you can edit `cwl/config.yml`.

Each backend defines its own `cut_dim` in `cwl/config.yml`, because different devices/samplers have different effective limits and embedding constraints.

Example:

```yml
input_matrix:
  class: File
  path: data/32x32_b.csv

cut_dim: 16
```

### 3. Executing the workflow

To run the full hybrid workflow:

```bash
streamflow run streamflow/streamflow.yml
```

On successful completion, Streamflow will output the `solutions.csv` file on the root of the project.

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
