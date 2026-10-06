# QSplit workflow catalogue

## Split and aggregate compatibility

The following table is the supported composition catalogue. Every marked pairing
can run without refinement or with any method in the next table. Choose one
initial pairing, then choose a compatible aggregator for consensus rounds.

Python paths below are relative to the `qsplit` package. All flat aggregators
accept `(solved_subproblems, global_qubo)`.

| Split function | Linear votes | Linear BP | Graph votes | Interactions weighting | Quadtree weighting | Recursive tree merge |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| `split_linear` | ✓ | ✓ | ✓ | — | — | — |
| `split_k_interactions` | ✓ | ✓ | ✓ | ✓ | — | — |
| `split_recursive_graph` | ✓ | ✓ | ✓ | — | — | — |
| `split_quadtree` | ✓ | ✓ | ✓ | — | ✓ | — |
| `split_recursive` | ✓ | ✓ | ✓ | — | — | ✓ |

Aggregator modules, in table order:

- `aggregate_linear`: count every binary local sample; ties choose zero.
- `aggregate_linear_belief_propagation`: normalize finite-energy sample weights inside each problem; ties choose one.
- `aggregate_recursive_graph`: use one best local sample; tied global votes keep the first usable value.
- `aggregate_k_interactions`: pipeline-specific positional centre weighting.
- `aggregate_quadtree`: vote only for exact real row IDs; the first row gets extra weight.
- `aggregate_recursive.aggregate_leaves`: requires the `split_tree` metadata created by `split_leaves` and the same leaf order.

A dash means the combination is outside this supported catalogue. A function may
accept the input shape while applying an inappropriate weighting or topology.
In particular, recursive `aggregate_solutions` expects exactly the three children
from **one** `split_recursive.` call. It cannot merge an arbitrary
flat list. `aggregate_leaves` performs the complete bottom-up reconstruction.
`aggregate_solutions_trivial` is for the two diagonal children after logical expansion.

## Refinement compatibility

The three independent methods are available on **all five pipelines**, including
linear BP and recursive logical expansion. `consensus` is a convenience selector.

| `REFINEMENT_METHOD` | Linear / BP | Recursive / logical expansion | Interactions | Graph | Quadtree |
| --- | --- | --- | --- | --- | --- |
| `conditioned` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `mean_field` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `soft_consensus` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `consensus` | Mean-field | Mean-field | Mean-field | Mean-field | Soft-consensus |

**Conditioned** operates on the global incumbent. Each sweep shuffles real
variables into bounded blocks, fixes the variables outside the block, solves it,
and accepts a whole local sample only if the original global energy does not
increase. Each accepted update is visible to the next block. No consensus
aggregator is needed after initialization.

**Mean-field** extracts real row and column windows from the preceding problems,
removes duplicate windows and padding, restores original internal interactions,
and adds fractional outside boundary fields from shared beliefs. Large windows
are divided to respect `CUT_DIM`. Macrovariables are discarded; quadtree's exact
centre windows still cover all real variables.

**Soft-consensus** rebuilds original principal objectives for real windows and
adds coefficient-scaled soft penalties towards shared beliefs. Rectangular blocks
become principal windows, and padding is removed. Quadtree macrovariable groups
retain their local membership metadata and compressed objective. Penalties are
rebuilt every round, never accumulated. Strength zero removes the guidance
penalties while leaving the loop enabled.

For real variable `i`, the shared belief is the mean of its global incumbent bit
and its mean estimate across the best finite-energy samples of each solved local
problem. Each local problem contributes once. Negative IDs are local to their
problem and are never combined across problems.
