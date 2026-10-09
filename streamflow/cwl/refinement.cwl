cwlVersion: v1.2
class: Workflow
requirements:
  ScatterFeatureRequirement: {}
  MultipleInputFeatureRequirement: {}
inputs:
  solver_configs:
    type: File[]
    default: []
  backend:
    type: string?
    default: null
  input_state: File
  running: boolean
outputs:
  state:
    type: File
    outputSource: update/state
  continue_refinement:
    type: boolean
    outputSource: update/continue_refinement
  solutions:
    type: File
    outputSource: update/solutions
  history:
    type: File
    outputSource: update/history
steps:
  prepare:
    run: clt/refinement_prepare.cwl
    in:
      input_state: input_state
      running: running
    out:
    - state
    - subproblems
    - solved_constant
  solve:
    run: clt/scatter.cwl
    in:
      input_qubo: prepare/subproblems
      configs: solver_configs
      backend: backend
    out:
    - solved_qubo
    scatter: input_qubo
  merge_solved:
    run: clt/merge_solved_lists.cwl
    in:
      parallel_solved: solve/solved_qubo
      split_solved: prepare/solved_constant
      iqm_solved:
        default: []
      quantinuum_h2_solved:
        default: []
      quantinuum_h2e_solved:
        default: []
    out:
    - solved_list
  update:
    run: clt/refinement_update.cwl
    in:
      input_state: prepare/state
      solved_list: merge_solved/solved_list
    out:
    - state
    - continue_refinement
    - solutions
    - history
