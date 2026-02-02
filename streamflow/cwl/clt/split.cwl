cwlVersion: v1.2
class: CommandLineTool

baseCommand: [python3, /workspace/qsplit/cwl/cli/split.py]

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
  cut_dim:
    type: int
    default: 10
    inputBinding: { prefix: --cut-dim }
  out_dir:
    type: string
    default: "subproblems"
    inputBinding: { prefix: --out-dir }

outputs:
  sub_qubos:
    type: File[]
    outputBinding:
      glob: "subproblems/*.pkl"

  solved_qubos:
    type: File[]?
    outputBinding:
      glob: "solved_dummy/*.pkl"

  full_qubo:
    type: File
    outputBinding: { glob: "initial_qubo.pkl" }

  tree_meta:
    type: File
    outputBinding: { glob: "tree.json" }
