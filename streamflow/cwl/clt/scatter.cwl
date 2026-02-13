cwlVersion: v1.2
class: CommandLineTool

baseCommand: [cli_scatter]

inputs:
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
