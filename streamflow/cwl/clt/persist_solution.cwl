cwlVersion: v1.2
class: CommandLineTool

baseCommand: [cli_persist_instance_solution]

inputs:
  input_solution:
    type: File
    inputBinding:
      prefix: --input-solution
  input_matrix:
    type: File
    inputBinding:
      prefix: --input-matrix
  solutions_dir:
    type: string
    default: solutions
    inputBinding:
      prefix: --solutions-dir

outputs:
  persisted_solution:
    type: File
    outputBinding:
      glob: "persisted_solution.csv"

arguments:
  - "--output-solution"
  - "persisted_solution.csv"
