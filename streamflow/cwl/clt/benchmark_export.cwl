cwlVersion: v1.2
class: CommandLineTool
requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing:
      - entryname: pipeline.json
        entry: |
          ${
            return JSON.stringify({
              split_method: inputs.split_method,
              aggregate_method: inputs.aggregate_method,
              cut_dim: inputs.cut_dim,
              refinement_method: inputs.refinement_method,
              refinement_loops: inputs.refinement_loops
            });
          }
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
  aggregate_method:
    type: string?
  cut_dim:
    type: int?
  refinement_method:
    type: string?
  refinement_loops:
    type: int?
arguments:
- --output-dir
- benchmark_results
- --pipeline-config
- pipeline.json
outputs:
  benchmark_results:
    type: Directory
    outputBinding:
      glob: benchmark_results
  report:
    type: File
    outputBinding:
      glob: benchmark_results/results.json
