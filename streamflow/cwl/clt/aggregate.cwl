cwlVersion: v1.2
class: CommandLineTool

baseCommand: [python3, -m, qsplit.cli.cli_aggregate]

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
    type: File[]
    inputBinding:
      prefix: --solved-list
      separate: true
      valueFrom: |
        ${
          return inputs.solved_list.map(f => f.path).join(",");
        }

outputs:
  aggregate_solutions:
    type: File
    outputBinding:
      glob: "aggregate.solutions.csv"

arguments:
requirements:
  - class: InlineJavascriptRequirement
