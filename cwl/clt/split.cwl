cwlVersion: v1.2
class: CommandLineTool

baseCommand: [python, -m, qsplit.cli_split]

inputs:
  input_qubo:
    type: File
    inputBinding:
      prefix: --input-matrix

  cut_dim:
    type: int
    inputBinding:
      prefix: --cut-dim

  approach:
    type: string
    inputBinding:
      prefix: --approach

  adaptive:
    type: boolean
    inputBinding:
      prefix: --adaptive

  out_dir:
    type: string
    inputBinding:
      prefix: --out-dir

outputs:
  sub_qubos:
    type: File[]
    outputBinding:
      glob: "subproblems/*.pkl"
      
  full_qubo:
    type: File
    outputBinding:
      glob: "initial_qubo.pkl"