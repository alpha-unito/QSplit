cwlVersion: v1.2
class: CommandLineTool

baseCommand: [cli_split]

inputs:
  barrier:
    type:
      - "null"
      - File[]
    default: null
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
  enable_sparse_check:
    type: boolean
    default: false
    inputBinding:
      prefix: --enable-sparse-check
  enable_iqm:
    type: boolean
    default: true
    inputBinding:
      prefix: --enable-iqm
  enable_quantinuum_h2:
    type: boolean
    default: false
    inputBinding:
      prefix: --enable-quantinuum-h2
  enable_quantinuum_h2e:
    type: boolean
    default: false
    inputBinding:
      prefix: --enable-quantinuum-h2e
  iqm_real_jobs:
    type: string
    default: "1"
    inputBinding:
      prefix: --iqm-real-jobs
  quantinuum_h2_real_jobs:
    type: string
    default: "1"
    inputBinding:
      prefix: --quantinuum-h2-real-jobs
  quantinuum_h2e_real_jobs:
    type: string
    default: "1"
    inputBinding:
      prefix: --quantinuum-h2e-real-jobs
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
    type: File[]
    outputBinding:
      glob: "solved_dummy/*.pkl"

  iqm_qubos:
    type: File[]
    outputBinding:
      glob: "planned/iqm/*.pkl"

  quantinuum_h2_qubos:
    type: File[]
    outputBinding:
      glob: "planned/quantinuum_h2/*.pkl"

  quantinuum_h2e_qubos:
    type: File[]
    outputBinding:
      glob: "planned/quantinuum_h2e/*.pkl"

  parallel_qubos:
    type: File[]
    outputBinding:
      glob: "planned/parallel/*.pkl"

  full_qubo:
    type: File
    outputBinding: { glob: "initial_qubo.pkl" }

  tree_meta:
    type: File
    outputBinding: { glob: "tree.json" }
