# Optional refinement

All QSplit pipelines support optional refinement: linear/iterative (including
belief propagation), recursive (including logical expansion), interactions,
graph partitioning and quadtree. The same native strategies can be reused across
pipelines. Start with the [workflow catalogue](workflows.md) for selection,
compatibility tables, entry points and runnable examples.

## Execution contract

Refined samplers first execute their initial split/solve/aggregate pipeline.
`qsplit.refinement.refine_result` can also start directly from an existing complete
global solution. It runs at most `REFINEMENT_LOOPS` additional rounds and returns
the best result encountered, evaluating every candidate using the original QUBO:

```text
energy = x.T @ Q_original @ x + original_offset
```

The initial objective is copied before a splitter can pad or modify it.
`result.refinement_history` records the initial global energy and each additional
round's energy, including worsening consensus rounds. Only improvements replace
the best solutions. Equal-energy conditioned moves can change the working state
without discarding the retained best result.

Missing, zero or negative loop budgets disable refinement. Refined samplers
then call their legacy counterpart directly, ignoring all other refinement
settings and callbacks. `refine_result` returns its input unchanged when disabled.
Legacy sampler functions continue to execute their original single pass.

A stalled round improves the best energy by at most `REFINEMENT_TOLERANCE`.
`REFINEMENT_PATIENCE` consecutive stalled rounds stop the loop. Defaults are ten
for conditioned sweeps and one for consensus/custom callbacks. The first extra
round does not require an improvement from initialization; there is no earlier
energy to compare with. All improvements are retained, even below the stopping
tolerance. An explicitly empty template list stops further rounds.

## Conditioned block refinement

`REFINEMENT_METHOD: conditioned` is the default. This shared C++ algorithm works
on real variables, independently of the original split and aggregate methods.
Every sweep shuffles variables with `REFINEMENT_SEED` and partitions them into
blocks no larger than `CUT_DIM` (the effective even dimension for the ready-to-run
quadtree sampler). Outside variables stay at their current incumbent values.

For a block `S`, boundary conditioning changes the local diagonal to:

```text
Q_local[i, i] = Q_original[i, i]
              + sum((Q_original[i, j] + Q_original[j, i]) * x_j for j outside S)
```

Internal interactions are copied from the original matrix. The offset includes
the fixed outside energy. A complete local sample therefore has exactly the
energy of the corresponding patched global assignment.

Each returned whole local sample is patched into the global state, evaluated
against the original objective, and accepted only if it does not increase global
energy. The next block immediately sees accepted changes. This avoids combining
simultaneous local improvements that conflict with one another. Different
partitions and patience allow exploration after stalled sweeps.

There is no hidden global annealing/exact pass in the refinement algorithm.
It calls the supplied solver on conditioned bounded blocks. Quadtree
macrovariables are used only during initialization; conditioned sweeps free their
individual real members.

## Shared beliefs

For every real variable, each preceding solved subproblem contributes the mean
binary value among its best finite-energy samples. Every subproblem contributes
once, independently of sample count or energy scale. Missing, non-binary and
non-finite-energy evidence is ignored. Negative IDs represent padding or local
macrovariables and are not matched across problems.

```text
belief_i = (best_global_assignment_i + mean(local_estimates_i)) / 2
```

Without usable local evidence, the belief is the global assignment. All beliefs
are collected before the next round is solved; propagation is synchronous.

## Mean-field boundary refinement

Select `REFINEMENT_METHOD: mean_field`, or reuse
`qsplit.refinement.refine_mean_field.refine_problems` as a callback.

The strategy extracts distinct real row and column windows from the preceding
problems. Padding and macrovariables are removed. Large windows are divided into
blocks no larger than `CUT_DIM`. Rectangular off-diagonal blocks become separate
principal objectives for both real windows, so a positional diagonal is never
mistaken for a real-variable linear term.

The conditioned diagonal formula above uses `belief_j` instead of `x_j`.
The local offset is the expected outside objective under independent Bernoulli
variables, including `E[x_j**2] = belief_j` for diagonal terms. Internal original
couplings are fully restored. The method approximates outside correlations and
then combines solved windows through a compatible flat aggregator.

The older `refine_linear.refine_problems` entry point uses the same numerical
implementation without slicing its existing linear windows. `consensus` on the
ready-to-run linear sampler retains that behaviour and its BP switch.

## Soft-consensus refinement

Select `REFINEMENT_METHOD: soft_consensus`, or reuse
`qsplit.refinement.refine_soft_consensus.refine_problems`.

Real row/column windows become bounded principal objectives. Compressed
quadtree templates retain exact variables and `macro_members`: each local
negative macrovariable maps to the real IDs it represents. Membership can differ
for the same negative ID in another template.

The strategy reconstructs the original local objective. For compressed groups,
this is `P.T @ Q_original @ P`, with `P` mapping real variables to exact or macro
representatives. It adds a soft binary-variable penalty:

```text
penalty_i = lambda_i * (x_i - target_i)**2
lambda_i = REFINEMENT_STRENGTH * sum(abs(local original coefficients involving i))
Q_local[i, i] += lambda_i * (1 - 2 * target_i)
offset_local += lambda_i * target_i**2
```

An exact variable targets its own belief. A macrovariable targets its members'
mean belief. A target of 0.5 gives no directional bias. Strength zero removes
penalties but leaves refinement enabled. No variable is fixed by the penalty.
Fields and penalties are always rebuilt from the original objective, so they
do not accumulate across rounds.

The older `refine_quadtree.refine_problems` retains the compressed-template
contract. `consensus` on the ready-to-run quadtree sampler selects that strategy.
On other ready-to-run samplers, `consensus` selects mean-field. The generic
composition/post-aggregation API chooses soft-consensus when macro templates
exist, and mean-field otherwise.

## Configuration, composition and orchestration

See [configuration and results](workflows.md#configuration-and-results) for all
YAML settings. Callbacks have the signature
`refinement(subproblems, original_qubo) -> list[QUBO]`. They receive previous
solved problems and the original objective with the best global solutions.
Treat inputs as read-only and return fresh objectives compatible with the
chosen refinement aggregator and backend limits.

The [composition API](workflows.md#compose-your-own-workflow) separates initial
aggregation from consensus aggregation. Recursive trees cannot be reused after
changing their leaf topology; use flat aggregation for new principal windows.
Conditioned sweeps require no subsequent consensus aggregation.

The shipped recursive CWL/StreamFlow workflows support an optional post-aggregation
refinement stage through `aggregate_configs`. It runs inside the aggregate process
using that process's solver configuration and writes the best result to the
existing CSV format. Additional rounds are synchronous backend calls, not new
scatter jobs. See [CWL and StreamFlow](workflows.md#cwl-and-streamflow-workflows).

## Validation and limits

These methods are heuristics. Small blocks can remain trapped, solvers can miss
improving samples, mean-field omits outside correlations, and macrovariables tie
multiple real variables to one bit. Consensus aggregation can combine incompatible
local proposals. The controller still retains the best evaluated global result.

The tests use independent exhaustive local solvers and real local simulated
annealing. They cover cross-pipeline reuse, independent aggregation, disabled
compatibility, offsets, padding, non-contiguous IDs, constant objectives, bounded
windows, original-energy evaluation, macrovariable-local beliefs, non-accumulating
penalties, sequential acceptance, logical expansion and CLI refinement.

```bash
uv run --no-sync python -m pytest \
  tests/unit/test_refinement.py tests/unit/test_refinement_workflows.py \
  tests/integration/test_local_runner.py tests/integration/test_cli_pipeline.py -q
```
