# QSplit

This repository contains a prototype of a **hybrid workflow for
Quadratic Unconstrained Binary Optimization (QUBO)** problems. The idea is to
show, in a concrete way, how a large QUBO can be split into smaller pieces,
solved on a quantum backend, and then combined again into a single,
consistent object.

### What this repository does

At a high level, the workflow does the following:

1. Takes the `cwl/data/input_matrix.csv` and convert it to a QUBO object, stored as a pickled `QUBO` object in `initial_qubo.pkl`.
2. Uses **QSplit** to decompose this matrix into several smaller sub-QUBOs.
3. Sends each sub-QUBO to a container that runs a D-Wave (or compatible)
   sampler and returns a solution.
4. Aggregates all the partial results into a new QUBO and a global assignment,
   stored in `aggregated.csv`.

The orchestration is expressed in **CWL** and executed with **Streamflow**, which
takes care of running the three stages (split, solve, aggregate) as separate
steps, possibly in parallel, inside Docker containers.

## Why QUBO splitting is useful today

Many modern optimisation and machine learning tasks can be written as QUBOs:
routing and scheduling problems, portfolio selection, some clustering and feature
selection formulations, and so on. In principle, these problems can be handed to
a quantum or quantum-inspired solver. In practice, two obstacles appear:

- current hardware has **limited size (e.g., few logical qubits) and non-trivial connectivity**, so a large
  dense QUBO does not fit directly on the device;
- even on classical hardware, solving a single large QUBO can be slow and
  difficult to scale.

QSplit sits between the model and the solver. It provides a controlled way to:

- cut a global QUBO into sub-problems that match the constraints of the target
  hardware (for example, a maximum number of variables);
- solve those sub-problems, possibly on different backends;
- rebuild a global view of the problem, including an updated QUBO and a global
  configuration.

The goal is not to claim optimality, but to offer a **reproducible and extensible
framework** to study QUBO decomposition strategies in a realistic, hybrid
setting.

## Software requirements

On the host machine you need:

- **Docker 4.24.2**, with the daemon running;
- **Streamflow**, installed in the Python environment:

```bash
pip install streamflow==0.2.0dev13
```

## Run

### 1. Building the Docker image

From the project root:

```bash
docker build -t qsplit:latest .
```

All steps in the workflow (split, solve, aggregate) share this image. This
choice emphasises reproducibility and makes the workflow easier to move across
machines or clusters.


### 2. Preparing the input QUBO

The repository ships an example matrix (`cwl/data/input_matrix.csv`) that will be QUBO serialised in `initial_qubo.pkl` during the first step. To
experiment with different inputs, you can edit `cwl/data/input_matrix.csv`.

The CWL configuration file `cwl/config.yml` references this file and
specifies parameters such as `cut_dim` and the output directory for sub‑problems.

### 3. Executing the workflow

To run the full hybrid workflow:

```bash
streamflow run streamflow.yml
```

Streamflow will:

- parse `cwl/main.cwl` and `cwl/config.yml`;
- deploy the containers defined in `streamflow.yml`;
- execute the `split`, `dwave_solve`, and `aggregate` steps;
- write the aggregated QUBO to the final matrix in a csv format.

On successful completion, Streamflow will output the `aggregated.csv` file on the root of the project.

