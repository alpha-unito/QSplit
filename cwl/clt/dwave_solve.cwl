cwlVersion: v1.2
class: CommandLineTool

baseCommand: [python3, -m, qsplit.cli_dwave_solve]

inputs:
  input_qubo:
    type: File
    inputBinding:
      prefix: --input-qubo

  backend:
    type: string
    inputBinding:
      prefix: --backend

outputs:
  solved_qubo:
    type: File
    outputBinding:
      glob: "solved.pkl"

arguments:
  - "--output-qubo"
  - "solved.pkl"