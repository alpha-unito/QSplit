cwlVersion: v1.2
class: CommandLineTool

baseCommand: [python3, -m, qsplit.cli.cli_split]

inputs:
  input_qubo:
    type: File
    inputBinding: { prefix: --input-matrix }
  cut_dim:
    type: int
    inputBinding: { prefix: --cut-dim }
  approach:
    type: string
    default: "dr"
    inputBinding: { prefix: --approach }
  adaptive:
    type: boolean
    inputBinding: { prefix: --adaptive }
  backends:
    type: string
    default: "dwave"
    inputBinding: { prefix: --backends }
  backend_cut_dims:
    type: string
    default: ""
    inputBinding: { prefix: --backend-cut-dims }
  out_dir:
    type: string
    default: "subproblems"
    inputBinding: { prefix: --out-dir }
  tree_file:
    type: string
    default: "tree.json"
    inputBinding: { prefix: --tree-file }
  backend_file:
    type: string
    default: "backends.json"
    inputBinding: { prefix: --backend-file }

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