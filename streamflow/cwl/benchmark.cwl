cwlVersion: v1.2
class: Workflow
requirements:
- class: SubworkflowFeatureRequirement
inputs:
  split_configs:
    type: File[]
    default: []
  parallel_configs:
    type: File[]
    default:
    - class: File
      location: benchmark.solver.yaml
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
  solutions_store_dir:
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
    default:
    - class: File
      location: benchmark.solver.yaml
  refinement_backend:
    type: string?
    default: null
  refinement_cut_dim:
    type: int?
    default: null
  benchmark_request: File
  qoblib_repository:
    type: Directory?
    default: null
outputs:
  benchmark_results:
    type: Directory
    outputSource: export/benchmark_results
  benchmark_bundle:
    type: Directory
    outputSource: prepare/benchmark_bundle
  report:
    type: File
    outputSource: export/report
  dataset_manifest:
    type: File
    outputSource: qsplit/dataset_manifest
  final_solutions:
    type: File[]
    outputSource: qsplit/final_solutions
  solutions_dir:
    type: Directory
    outputSource: qsplit/solutions_dir
  solutions_manifest:
    type: File
    outputSource: qsplit/solutions_manifest
  refinement_histories:
    type: File[]
    outputSource: qsplit/refinement_histories
  final_states:
    type: File[]
    outputSource: qsplit/final_states
steps:
  prepare:
    run: clt/benchmark_prepare.cwl
    in:
      request: benchmark_request
      repository: qoblib_repository
    out:
    - dataset
    - benchmark_bundle
  qsplit:
    run: main.cwl
    in:
      split_configs: split_configs
      parallel_configs: parallel_configs
      iqm_configs: iqm_configs
      quantinuum_h2_configs: quantinuum_h2_configs
      quantinuum_h2e_configs: quantinuum_h2e_configs
      aggregate_configs: aggregate_configs
      storage_configs: storage_configs
      cut_dim: cut_dim
      enable_sparse_check: enable_sparse_check
      enable_iqm: enable_iqm
      enable_quantinuum_h2: enable_quantinuum_h2
      enable_quantinuum_h2e: enable_quantinuum_h2e
      iqm_real_jobs: iqm_real_jobs
      quantinuum_h2_real_jobs: quantinuum_h2_real_jobs
      quantinuum_h2e_real_jobs: quantinuum_h2e_real_jobs
      solutions_store_dir: solutions_store_dir
      split_method: split_method
      aggregate_method: aggregate_method
      refinement_method: refinement_method
      refinement_loops: refinement_loops
      refinement_aggregate_method: refinement_aggregate_method
      refinement_configs: refinement_configs
      refinement_solver_configs: refinement_solver_configs
      refinement_backend: refinement_backend
      refinement_cut_dim: refinement_cut_dim
      dataset: prepare/dataset
    out:
    - dataset_manifest
    - final_solutions
    - solutions_dir
    - solutions_manifest
    - refinement_histories
    - final_states
  export:
    run: clt/benchmark_export.cwl
    in:
      bundle: prepare/benchmark_bundle
      solutions_dir: qsplit/solutions_dir
      split_method: split_method
      aggregate_method: aggregate_method
      cut_dim: cut_dim
      refinement_method: refinement_method
      refinement_loops: refinement_loops
    out:
    - benchmark_results
    - report
