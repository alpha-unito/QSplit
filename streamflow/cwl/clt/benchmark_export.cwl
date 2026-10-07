cwlVersion: v1.2
class: CommandLineTool
baseCommand:
- python
- -m
- qsplit.benchmark.cli
- export
inputs:
  bundle:
    type: Directory
    inputBinding:
      prefix: --bundle
  solutions_dir:
    type: Directory
    inputBinding:
      prefix: --solutions-dir
  split_method:
    type: string?
    inputBinding:
      prefix: --split-method
  aggregate_method:
    type: string?
    inputBinding:
      prefix: --aggregate-method
  cut_dim:
    type: int?
    inputBinding:
      prefix: --cut-dim
  refinement_method:
    type: string?
    inputBinding:
      prefix: --refinement-method
  refinement_loops:
    type: int?
    inputBinding:
      prefix: --refinement-loops
arguments:
- --output-dir
- benchmark_results
outputs:
  benchmark_results:
    type: Directory
    outputBinding:
      glob: benchmark_results
  report:
    type: File
    outputBinding:
      glob: benchmark_results/results.json
