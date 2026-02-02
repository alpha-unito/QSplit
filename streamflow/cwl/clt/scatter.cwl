cwlVersion: v1.2
class: CommandLineTool

baseCommand: [python3, /workspace/qsplit/cwl/cli/scatter.py]

inputs:
  input_qubo:
    type: File
    inputBinding:
      prefix: --input-qubo

  backend:
    type: string
    default: "auto"
    inputBinding:
      prefix: --backend

outputs:
  solved_qubo:
    type: File?
    outputBinding:
      glob: "solved.pkl"

arguments:
  - "--output-qubo"
  - "solved.pkl"
