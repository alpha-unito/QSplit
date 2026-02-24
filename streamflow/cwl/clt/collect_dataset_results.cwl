cwlVersion: v1.2
class: CommandLineTool

baseCommand: [cli_collect_dataset_results]

inputs:
  dataset_manifest:
    type: File
    inputBinding:
      prefix: --dataset-manifest
  solutions_dir:
    type: string
    default: solutions
    inputBinding:
      prefix: --solutions-dir
  solution_csv_list:
    type: File[]
    default: []
    inputBinding:
      prefix: --solution-csv
      separate: true

outputs:
  results_dir:
    type: Directory
    outputBinding:
      glob: "solutions_dataset"
  results_manifest:
    type: File
    outputBinding:
      glob: "dataset_results_manifest.json"

arguments:
  - "--output-dir"
  - "solutions_dataset"
  - "--output-manifest"
  - "dataset_results_manifest.json"
