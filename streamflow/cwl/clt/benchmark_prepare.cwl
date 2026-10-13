cwlVersion: v1.2
class: CommandLineTool
baseCommand:
- python
- -m
- qsplit.benchmark.cli
- prepare
inputs:
  request:
    type: File
    inputBinding:
      prefix: --request
  repository:
    type: Directory?
    inputBinding:
      prefix: --repository
arguments:
- --output-dir
- benchmark_bundle
outputs:
  dataset:
    type: File
    outputBinding:
      glob: benchmark_bundle/dataset.jsonl
  benchmark_bundle:
    type: Directory
    outputBinding:
      glob: benchmark_bundle
