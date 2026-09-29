# Optional local refinement

Refinement is available in `qsplit/local_runner.py` for iterative and quadtree
splitting. CWL and StreamFlow orchestration are unchanged.

The new entry points are:

- `qsplit_sampler_refined_iterative(qubo, *, refinement=None)`
- `qsplit_sampler_refined_quadtree(qubo, *, refinement=None)`

The default method is **conditioned block refinement**. The earlier mean-field
and soft-consensus methods remain available through `REFINEMENT_METHOD=consensus`.

The existing samplers keep their previous behavior, even when refinement
YAML settings are supplied. The local runner still uses its configured
`solve` function, currently D-Wave simulated annealing.

## Workflow

Initial splitting and aggregation remain unchanged. Refinement consumes their
global result. The conditioned method then composes local updates by their actual
global energy; the consensus method continues to use the original vote aggregator.

```text
split -> solve -> aggregate -> global incumbent
                                    |
                 +------------------+------------------+
                 |                                     |
         conditioned method                     consensus method
   condition block on current state           propagate local beliefs
                 |                                     |
             solve block                          solve all blocks
                 |                                     |
   accept a non-worsening global update             vote aggregate
                 |                                     |
       next block, then next sweep             evaluate global energy
                 +------------------+------------------+
                                    |
                    check budget and stalled rounds
```

1. Split and solve the input using the selected pipeline.
2. Aggregate the local samples into a global assignment. Iterative refinement
   respects the existing `local_runner.BP` switch; quadtree uses its existing
   weighted aggregation.
3. Recompute global energy as `x.T @ Q @ x + offset`, using the original matrix,
   original variable order, and original offset. Keep the best result.
4. Run a refinement sweep or consensus round if its budget remains.
5. Stop after `REFINEMENT_PATIENCE` consecutive rounds whose global improvement
   is at most the tolerance, or after exhausting the budget. Return the best result
   encountered, including the initial pass. Smaller improvements are also retained.

The first refinement round runs without requiring an improvement from the
initial pass: there is no earlier energy to compare against. A stalled round can
be followed by different neighborhoods and further improvements; it does not prove
convergence. An empty list of subproblems stops immediately.

Local energies after refinement belong to modified objectives. They are used to
select local samples, but never compared directly with global energies.
All built-in strategies rebuild from the original objective, so
fields and penalties do not accumulate. The iterative splitter can still pad
the returned QUBO in place, as it did previously; the energy reference is copied
before splitting.

## Configuration

| YAML key | Default | Meaning |
| --- | --- | --- |
| `REFINEMENT_LOOPS` | `0` | Maximum **additional** solve/aggregate rounds. Missing, zero, or negative values delegate directly to the corresponding legacy sampler. |
| `REFINEMENT_METHOD` | `conditioned` | Sequential block updates, or `consensus` for the earlier pipeline-specific propagation. |
| `REFINEMENT_PATIENCE` | `10` / `1` | Stalled rounds before stopping: default 10 for conditioned refinement, 1 for consensus or a custom callback. Must be positive. |
| `REFINEMENT_SEED` | `0` | Non-negative seed for conditioned neighborhood selection. Backend randomness is configured separately. |
| `REFINEMENT_TOLERANCE` | `1e-9` | Absolute global energy improvement required to continue. Must be finite and non-negative. |
| `REFINEMENT_STRENGTH` | `0.1` | Relative soft-penalty strength for quadtree consensus. Must be finite and non-negative. Not used by conditioned or linear refinement. |

`REFINEMENT_LOOPS=3` allows an initial pass and at most three additional sweeps
or consensus rounds. Stopping may occur earlier. Setting quadtree strength to zero removes its
guidance penalties; it does not disable the loop. Use `REFINEMENT_LOOPS=0` for
the exact previous execution path. Other refinement settings and custom
refinement callbacks are ignored when the loop is disabled.

Existing splitting parameters retain their meanings: `CUT_DIM` and `STRIDE` for
iterative splitting, and `CUT_DIM` and `EXACT_RATIO` for quadtree splitting.
Use a positive `CUT_DIM` (at least 2 for quadtree); quadtree rounds it down to an
even number. The strategies do not enlarge local problems beyond their split
window sizes.

## Usage

With the project's D-Wave dependencies installed:

```python
from qsplit import configuration

import numpy as np

from qsplit.local_runner import (
    qsplit_sampler_refined_iterative,
    qsplit_sampler_refined_quadtree,
)
from qsplit.qubo import QUBO

# configs/refinement.yaml contains CUT_DIM: 2, REFINEMENT_LOOPS: 3,
# and REFINEMENT_TOLERANCE: 1.0e-9 (one key per YAML line).
config = configuration.load("configs/refinement.yaml")

matrix = np.array(
    [
        [-2.0, 1.0, -3.0, 0.0],
        [0.0, -1.0, 2.0, -2.0],
        [0.0, 0.0, 1.0, -3.0],
        [0.0, 0.0, 0.0, -1.0],
    ]
)
ids = np.arange(len(matrix))

# Use separate inputs because splitting can pad a QUBO in place.
iterative = qsplit_sampler_refined_iterative(QUBO(matrix.copy(), ids.copy(), ids.copy()), config=config)
quadtree = qsplit_sampler_refined_quadtree(QUBO(matrix.copy(), ids.copy(), ids.copy()), config=config)

print(iterative.solutions)
print(iterative.refinement_history)
print(quadtree.solutions)

# The same entry point also supports the previous single-pass behavior.
config["REFINEMENT_LOOPS"] = 0
single_pass = qsplit_sampler_refined_iterative(QUBO(matrix.copy(), ids.copy(), ids.copy()), config=config)
```

For enabled runs, `result.refinement_history` records each round's global energy,
starting with the initial pass. Conditioned histories are non-increasing;
consensus histories can include rejected, worsening rounds. `result.solutions` contains the best round's
solutions. The number of executed refinement rounds is
`len(result.refinement_history) - 1`. Disabled runs follow the old return contract
and do not create this attribute.

## Conditioned block refinement

This method uses exact binary boundary conditions, rather than mean-field beliefs:
when solving variables in a block `S`, every variable outside `S` is held at its
value in the current global assignment. The local diagonal becomes:

```text
Q_local[i, i] = Q[i, i] + sum((Q[i, j] + Q[j, i]) * x_j for j outside S)
```

The local offset includes the fixed outside energy. Therefore a complete local
assignment has exactly the energy of the corresponding patched global assignment.
All coefficients come from the original QUBO; there are no artificial consensus
penalties. `condition_subproblem` implements both binary conditioning and the
fractional variant used by linear consensus.

Each sweep shuffles the real variables and partitions them into blocks no larger
than `CUT_DIM` (the even effective size for quadtree). It solves each block,
inserts each returned **whole local sample** into the global state, and accepts
only non-worsening candidates after evaluating the original objective. The next
block sees the updated assignment immediately. Equal-energy moves may change
the working state, while the best result is retained separately.

Changing the partition between sweeps lets variables previously separated by a
cut move together. Sequential updates avoid combining locally beneficial but
mutually incompatible changes. Patience allows stalled sweeps to be followed by
different neighborhoods. The backend is still `local_runner.solve`; there is no
hidden global SA pass or exact optimizer in the refinement algorithm.

Both pipelines use this same refinement method after their different initial
split/solve/aggregate passes. Quadtree macrovariables are used in initialization;
subsequent conditioned sweeps work on real variables. This deliberately avoids
forcing a group of real variables to keep sharing one macrovariable bit.

This is a block-coordinate heuristic, not a guarantee of global optimality.
It can remain trapped when an improving move requires more variables than fit
in a block, or when the backend misses an improving sample. The use of a global
incumbent and conditioned subproblems follows the general decomposition pattern
documented in [D-Wave's decomposition workflows](https://docs.dwavequantum.com/en/latest/quantum_research/decomposing.html).

## Consensus mode: propagated information

For each real variable `i`, each subproblem containing it contributes the mean
binary value among its best finite-energy samples. Each subproblem has equal
weight regardless of its sample count or energy scale. Missing, non-binary,
and non-finite-energy evidence is ignored. Negative IDs are never matched
between subproblems: they represent padding or local macrovariables.

The belief used for propagation is:

```text
p_i = (global_assignment_i + mean(local_estimates_i)) / 2
```

If no usable local estimate exists, `p_i` is the global assignment. Thus a
variable shared by several subproblems conveys their combined evidence, while
the global solution supplies context for variables outside a local problem.
Beliefs are collected from the completed round before any new problem is solved;
propagation is synchronous, independent of solve order.

### Iterative: restore boundary interactions

Linear splitting produces both diagonal and off-diagonal matrix blocks.
Off-diagonal blocks have different row and column IDs, so adding a bias to
their positional diagonal would not represent a linear term for a real variable.

The first refinement converts the distinct real row/column windows into
principal sub-QUBOs extracted from the original matrix. Duplicate windows are
removed and padding is excluded. Both sides of off-diagonal blocks are retained.
This can reduce the number of local problems relative to the initial split.
Subsequent rounds reuse these windows.

Within a window `S`, all original internal interactions are restored. Outside
variables contribute an expected boundary field to each local diagonal:

```text
Q_local[i, i] = Q[i, i] + sum((Q[i, j] + Q[j, i]) * p_j for j outside S)
```

This is a mean-field approximation: external variables are treated as independent
Bernoulli variables with means `p_j`. Their expected energy is included in the
local offset, using `E[x_j²] = p_j` for diagonal terms. Local variables remain
free binary decisions. For example, an external variable likely to be 1 sends a
negative field through a negative coupling, encouraging the local variable to
become 1 as well.

### Quadtree: consensus for exact and macrovariables

The quadtree splitter records `subproblem.macro_members`, mapping each local
negative macrovariable ID to the real IDs it represents. The same negative ID
in another subproblem can have entirely different members.

Each round reconstructs the compressed original objective, keeping the existing
exact variables and macrovariable groups. In matrix form this is `P.T @ Q @ P`,
where `P` maps each real variable to its exact or macrovariable representative.
It then adds a soft penalty for each local variable:

```text
penalty = lambda_i * (x_i - target_i)**2
lambda_i = REFINEMENT_STRENGTH * sum(abs(original local coefficients involving i))
```

An exact variable's target is its propagated belief. A macrovariable's target
is the mean belief of its real members. Because `x_i` is binary, the penalty
adds `lambda_i * (1 - 2 * target_i)` to its diagonal and
`lambda_i * target_i**2` to the offset. A target of 0.5 gives no directional bias.
Variables are encouraged toward consensus, never fixed.

Enabled runs treat macrovariables as active variables when checking whether a
local problem is empty. Constant local objectives, including ones created by
field cancellation, receive an all-zero assignment with their offset as energy
without calling the backend. This keeps undefined dummy samples out of aggregation.

## Adding a refinement method

Implement the same function shape as the built-in strategies:

```python
def refine_problems(subproblems: list[QUBO], qubo: QUBO) -> list[QUBO]: ...
```

`subproblems` contains the previous round's solved problems. `qubo` contains the
original objective and the best global solutions so far. Treat these inputs as
read-only and return fresh QUBOs ready for solving. Preserve the information
needed by subsequent rounds, such as `macro_members` for quadtree. Return
problems compatible with the pipeline's aggregator and backend size limit.

Select a custom consensus-style method through the sampler's keyword argument;
an explicit callback takes precedence over `REFINEMENT_METHOD`:

```python
result = qsplit_sampler_refined_iterative(qubo, refinement=my_refine_problems)
```

`collect_beliefs` in `qsplit/refinement/propagation.py` can be reused by additional
methods. The loop controller remains responsible for solving, aggregating,
measuring global improvement, stopping, and retaining the best solution.

## Limits and validation

These methods are heuristics. In consensus mode, mean-field propagation omits
correlations between external variables, and a macrovariable still represents
several real variables with one bit. Simultaneous proposals and majority voting
can also destroy correlations or combine incompatible local improvements.
Conditioned refinement avoids those issues, but cannot guarantee a global
optimum with bounded blocks and a heuristic solver. Exhausting patience is a
stopping rule, not a proof of convergence. Stochastic backends can produce
different histories on different runs.

The knapsack QUBO includes a squared capacity/slack penalty. A feasible item
selection can still have high QUBO energy when its slack bits are inconsistent.
The earlier quadtree consensus penalty is scaled by QUBO coefficient magnitudes,
which can overwhelm small item-value differences on heavily penalized instances.
Increasing the number of consensus loops alone does not fix those problems.

The comparison script reports exact knapsack value and percentage gap, computed
by dynamic programming **only as a benchmark reference**, and the residual QUBO
penalty (`energy + value`). It also reports solver calls and total reads.
Optionally compare direct SA at the largest pipeline's total read budget:

```bash
python -m sketch.compare_knapsack_refinement --size 20 --cut-dim 10 \
    --refinement-loops 25 --seed 42 --match-sa-budget
```

Use `--refinement-method consensus` to reproduce the earlier algorithm. Increase
`--refinement-loops` and `--patience` together for longer plateau exploration.
Equal read counts are not equal computational work: a direct SA read updates the
entire problem, whereas a local read updates a block. Compare both quality and
elapsed time, and use multiple seeds.

`tests/test_refinement.py` checks disabled behavior, loop limits, stopping,
best-result retention, original-energy evaluation, shared-variable propagation,
macrovariable locality, non-accumulating penalties, padding, non-contiguous
variable IDs, constant problems, and custom methods. Conditioned tests check
global/local energy equivalence, joint moves across single-variable barriers,
safe sequential composition, changing bounded neighborhoods, rejection of bad
sampler results, and patience. Deterministic exact local solves also demonstrate
improvement on one instance for each consensus pipeline.

```bash
python -m pytest tests/test_refinement.py -q
```
