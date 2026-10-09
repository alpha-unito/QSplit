cwlVersion: v1.2
class: CommandLineTool

baseCommand: [cli_dataset_prepare]

inputs:
  configs:
    type:
      type: array
      items: File
      inputBinding:
        prefix: --config
    default: []
    inputBinding: {}
  dataset_jsonl:
    type: File
    inputBinding:
      prefix: --dataset-jsonl
  max_instances:
    type: int
    default: 0
    inputBinding:
      prefix: --max-instances
  solutions_dir:
    type: string
    default: solutions
    inputBinding:
      prefix: --solutions-dir

outputs:
  matrix_files:
    type: File[]
    outputBinding:
      glob: "dataset_matrices/*.csv"
  dataset_manifest:
    type: File
    outputBinding:
      glob: "dataset_manifest.json"

arguments:
  - "--output-dir"
  - "dataset_matrices"
  - "--manifest"
  - "dataset_manifest.json"
