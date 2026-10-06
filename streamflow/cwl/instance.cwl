cwlVersion: v1.2
class: Workflow
requirements:
- class: ScatterFeatureRequirement
- class: MultipleInputFeatureRequirement
- class: SubworkflowFeatureRequirement
- class: InlineJavascriptRequirement
inputs:
  split_configs:
    type: File[]
    default: []
  parallel_configs:
    type: File[]
    default: []
  iqm_configs:
    type: File[]
    default: []
  quantinuum_h2_configs:
    type: File[]
    default: []
  quantinuum_h2e_configs:
    type: File[]
    default: []
  aggregate_configs:
    type: File[]
    default: []
  storage_configs:
    type: File[]
    default: []
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
    default: '1'
  quantinuum_h2_real_jobs:
    type: string
    default: '1'
  quantinuum_h2e_real_jobs:
    type: string
    default: '1'
  solutions_dir:
    type: string
    default: solutions
  split_method:
    type: string
    default: recursive
  aggregate_method:
    type: string
    default: auto
  refinement_method:
    type: string
    default: none
  refinement_loops:
    type: int
    default: 0
  refinement_aggregate_method:
    type: string
    default: linear
  refinement_configs:
    type: File[]
    default: []
  refinement_solver_configs:
    type: File[]
    default: []
  refinement_backend:
    type: string?
    default: null
  refinement_cut_dim:
    type: int?
    default: null
outputs:
  final_solutions:
    type: File
    outputSource: persist_solution/persisted_solution
  final_state:
    type: File
    outputSource:
    - refine/state
    - initialize_refinement/state
    pickValue: first_non_null
  final_history:
    type: File
    outputSource:
    - refine/history
    - initialize_refinement/history
    pickValue: first_non_null
steps:
  split:
    run: clt/split.cwl
    in:
      configs: split_configs
      input_qubo: input_matrix
      adaptive:
        default: true
      cut_dim: cut_dim
      enable_sparse_check: enable_sparse_check
      enable_iqm: enable_iqm
      enable_quantinuum_h2: enable_quantinuum_h2
      enable_quantinuum_h2e: enable_quantinuum_h2e
      iqm_real_jobs: iqm_real_jobs
      quantinuum_h2_real_jobs: quantinuum_h2_real_jobs
      quantinuum_h2e_real_jobs: quantinuum_h2e_real_jobs
      split_method: split_method
      aggregate_method: aggregate_method
    out:
    - sub_qubos
    - solved_qubos
    - full_qubo
    - tree_meta
    - iqm_qubos
    - quantinuum_h2_qubos
    - quantinuum_h2e_qubos
    - parallel_qubos
  parallelize:
    run: clt/scatter.cwl
    in:
      configs: parallel_configs
      input_qubo: split/parallel_qubos
    out:
    - solved_qubo
    scatter:
    - input_qubo
  iqm:
    run: clt/scatter.cwl
    in:
      backend:
        default: iqm
      configs: iqm_configs
      input_qubo: split/iqm_qubos
    out:
    - solved_qubo
    scatter:
    - input_qubo
  quantinuum_h2:
    run: clt/scatter.cwl
    in:
      backend:
        default: quantinuum_h2
      configs: quantinuum_h2_configs
      input_qubo: split/quantinuum_h2_qubos
    out:
    - solved_qubo
    scatter:
    - input_qubo
  quantinuum_h2e:
    run: clt/scatter.cwl
    in:
      backend:
        default: quantinuum_h2e
      configs: quantinuum_h2e_configs
      input_qubo: split/quantinuum_h2e_qubos
    out:
    - solved_qubo
    scatter:
    - input_qubo
  merge_solved:
    run: clt/merge_solved_lists.cwl
    in:
      parallel_solved: parallelize/solved_qubo
      iqm_solved: iqm/solved_qubo
      quantinuum_h2_solved: quantinuum_h2/solved_qubo
      quantinuum_h2e_solved: quantinuum_h2e/solved_qubo
      split_solved: split/solved_qubos
    out:
    - solved_list
  aggregate:
    run: clt/aggregate.cwl
    in:
      configs: aggregate_configs
      input_qubo: split/full_qubo
      tree_file: split/tree_meta
      solved_list: merge_solved/solved_list
      aggregate_method: aggregate_method
      skip_local_refinement:
        default: true
    out:
    - aggregate_solutions
    - aggregate_qubo
  initialize_refinement:
    run: clt/refinement_initialize.cwl
    in:
      configs: refinement_configs
      input_state: aggregate/aggregate_qubo
      method: refinement_method
      loops: refinement_loops
      cut_dim:
        source:
        - refinement_cut_dim
        - cut_dim
        pickValue: first_non_null
      aggregate_method: refinement_aggregate_method
    out:
    - state
    - continue_refinement
    - solutions
    - history
  refine:
    run: refinement.cwl
    requirements:
      http://commonwl.org/cwltool#Loop:
        loopWhen: $(inputs.running)
        loop:
          input_state:
            loopSource: state
          running:
            loopSource: continue_refinement
        outputMethod: last
    in:
      solver_configs: refinement_solver_configs
      backend: refinement_backend
      input_state: initialize_refinement/state
      running: initialize_refinement/continue_refinement
    out:
    - state
    - continue_refinement
    - solutions
    - history
  persist_solution:
    run: clt/persist_solution.cwl
    in:
      configs: storage_configs
      input_solution:
        source:
        - refine/solutions
        - aggregate/aggregate_solutions
        pickValue: first_non_null
      input_matrix: input_matrix
      solutions_dir: solutions_dir
    out:
    - persisted_solution
