cwlVersion: v1.2
class: CommandLineTool

baseCommand: [python3, /workspace/qsplit/cwl/cli/aggregate.py]

inputs:
  input_qubo:
    type: File
    inputBinding:
      prefix: --input-qubo
      separate: true

  tree_file:
    type: File
    inputBinding:
      prefix: --tree-file
      separate: true

  solved_list:
    type: File[]?
    default: []
    inputBinding:
      prefix: --solved-list
      separate: true

outputs:
  aggregate_solutions:
    type: File
    outputBinding:
      glob: "solutions.csv"

arguments:
requirements: []
