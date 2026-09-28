cwlVersion: v1.2
class: CommandLineTool

baseCommand: [cli_scatter]

inputs:
  configs:
    type:
      type: array
      items: File
      inputBinding:
        prefix: --config
    default: []
    inputBinding: {}
  backend:
    type: string?
    inputBinding: { prefix: --backend }
  input_qubo:
    type: File
    inputBinding:
      prefix: --input-qubo

outputs:
  solved_qubo:
    type: File
    outputBinding:
      glob: "solved.pkl"

arguments:
  - "--output-qubo"
  - "solved.pkl"
