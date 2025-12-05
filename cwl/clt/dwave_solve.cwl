cwlVersion: v1.2
class: CommandLineTool

baseCommand: [python, -m, qsplit.cli_dwave_solve]

inputs:
  input_qubo:
    type: File
    inputBinding:
      prefix: --input-qubo

  backend:
    type: string

  output_qubo_name:
    type: string
    default: solved.pkl
    inputBinding:
      prefix: --output-qubo

outputs:
  solved_qubo:
    type: File
    outputBinding:
      glob: $(inputs.output_qubo_name)