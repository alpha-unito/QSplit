cwlVersion: v1.2
class: CommandLineTool

baseCommand: [python3, -m, qsplit.cli.cli_split]

inputs:
  input_qubo:
    type: File
    inputBinding: { prefix: --input-matrix }
  approach:
    type: string
    default: "dr"
    inputBinding: { prefix: --approach }
  adaptive:
    type: boolean
    inputBinding: { prefix: --adaptive }
  backend_cut_dims:
    type: string
    default: ""
    inputBinding: { prefix: --backend-cut-dims }
  out_dir:
    type: string
    default: "subproblems"
    inputBinding: { prefix: --out-dir }

outputs:
  sub_qubos:
    type: File[]
    outputBinding:
      glob: "subproblems/*/*.pkl"

  full_qubo:
    type: File
    outputBinding: { glob: "initial_qubo.pkl" }

  tree_meta:
    type: File
    outputBinding: { glob: "tree.json" }