cwlVersion: v1.2
class: Workflow

requirements:
  - class: ScatterFeatureRequirement
  - class: MultipleInputFeatureRequirement

inputs:
  input_matrix: File
  cut_dim: int
  enable_sparse_check:
    type: boolean
    default: false
  enable_iqm:
    type: boolean
    default: false
  enable_quantinuum_h2:
    type: boolean
    default: false
  enable_quantinuum_h2e:
    type: boolean
    default: false
  iqm_real_jobs:
    type: string
    default: "1"
  quantinuum_h2_real_jobs:
    type: string
    default: "1"
  quantinuum_h2e_real_jobs:
    type: string
    default: "1"
  solutions_dir:
    type: string
    default: solutions

outputs:
  final_solutions:
    type: File
    outputSource: persist_solution/persisted_solution

steps:
  split:
    run: clt/split.cwl
    in:
      input_qubo: input_matrix
      adaptive: { default: true }
      cut_dim: cut_dim
      enable_sparse_check: enable_sparse_check
      enable_iqm: enable_iqm
      enable_quantinuum_h2: enable_quantinuum_h2
      enable_quantinuum_h2e: enable_quantinuum_h2e
      iqm_real_jobs: iqm_real_jobs
      quantinuum_h2_real_jobs: quantinuum_h2_real_jobs
      quantinuum_h2e_real_jobs: quantinuum_h2e_real_jobs
    out:
      [
        sub_qubos,
        solved_qubos,
        full_qubo,
        tree_meta,
        iqm_qubos,
        quantinuum_h2_qubos,
        quantinuum_h2e_qubos,
        parallel_qubos,
      ]

  parallelize:
    run: clt/scatter.cwl
    in:
      input_qubo: split/parallel_qubos
    out: [solved_qubo]
    scatter: [input_qubo]

  iqm:
    run: clt/scatter.cwl
    in:
      input_qubo: split/iqm_qubos
    out: [solved_qubo]
    scatter: [input_qubo]

  quantinuum_h2:
    run: clt/scatter.cwl
    in:
      input_qubo: split/quantinuum_h2_qubos
    out: [solved_qubo]
    scatter: [input_qubo]

  quantinuum_h2e:
    run: clt/scatter.cwl
    in:
      input_qubo: split/quantinuum_h2e_qubos
    out: [solved_qubo]
    scatter: [input_qubo]

  merge_solved:
    run: clt/merge_solved_lists.cwl
    in:
      parallel_solved: parallelize/solved_qubo
      iqm_solved: iqm/solved_qubo
      quantinuum_h2_solved: quantinuum_h2/solved_qubo
      quantinuum_h2e_solved: quantinuum_h2e/solved_qubo
      split_solved: split/solved_qubos
    out: [solved_list]

  aggregate:
    run: clt/aggregate.cwl
    in:
      input_qubo: split/full_qubo
      tree_file: split/tree_meta
      solved_list: merge_solved/solved_list
    out: [aggregate_solutions]

  persist_solution:
    run: clt/persist_solution.cwl
    in:
      input_solution: aggregate/aggregate_solutions
      input_matrix: input_matrix
      solutions_dir: solutions_dir
    out: [persisted_solution]
