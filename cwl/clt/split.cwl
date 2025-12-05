cwlVersion: v1.2
class: CommandLineTool

requirements:
  - class: InlineJavascriptRequirement

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
    default: "dr"
    inputBinding:
      prefix: --approach

  adaptive:
    type: boolean
    inputBinding:
      prefix: --adaptive

  tree_file:
    type: string
    default: "tree.json"
    inputBinding:
      prefix: --tree-file

  out_dir:
    type: string
    default: "subproblems"
    inputBinding:
      prefix: --out-dir

  backends:
    type: string
    default: "dwave"
    inputBinding:
      prefix: --backends

  backend_cut_dims:
    type: string
    default: ""
    inputBinding:
      prefix: --backend-cut-dims

  backend_file:
    type: string
    default: "backends.json"
    inputBinding:
      prefix: --backend-file

outputs:
  sub_qubos:
    type: File[]
    outputBinding:
      glob: $(inputs.out_dir + "/*.pkl")

  full_qubo:
    type: File
    outputBinding:
      glob: "initial_qubo.pkl"

  tree_meta:
    type: File
    outputBinding:
      glob: $(inputs.tree_file)

  sub_backends:
    type: string[]
    outputBinding:
      glob: $(inputs.backend_file)
      loadContents: true
      outputEval: |
        ${
          return JSON.parse(self[0].contents);
        }