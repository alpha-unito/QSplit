cwlVersion: v1.2
class: CommandLineTool
baseCommand:
- cli_aggregate
inputs:
  configs:
    type:
      type: array
      items: File
      inputBinding:
        prefix: --config
    default: []
    inputBinding: {}
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
  aggregate_method:
    type: string
    default: auto
    inputBinding:
      prefix: --aggregate-method
  skip_local_refinement:
    type: boolean
    default: false
    inputBinding:
      prefix: --skip-local-refinement
outputs:
  aggregate_solutions:
    type: File
    outputBinding:
      glob: solutions.csv
  aggregate_qubo:
    type: File
    outputBinding:
      glob: aggregate_qubo.pkl
arguments: null
requirements: []
