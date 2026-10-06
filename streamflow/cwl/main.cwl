cwlVersion: v1.2
class: Workflow
requirements:
- class: ScatterFeatureRequirement
- class: MultipleInputFeatureRequirement
- class: SubworkflowFeatureRequirement
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
  dataset: File
  max_instances:
    type: int
    default: 0
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
    default: []
  refinement_backend:
    type: string?
    default: null
  refinement_cut_dim:
    type: int?
    default: null
outputs:
  dataset_manifest:
    type: File
    outputSource: prepare_dataset/dataset_manifest
  final_solutions:
    type: File[]
    outputSource: qsplit_instances/final_solutions
  solutions_dir:
    type: Directory
    outputSource: collect_results/results_dir
  solutions_manifest:
    type: File
    outputSource: collect_results/results_manifest
  refinement_histories:
    type: File[]
    outputSource: qsplit_instances/final_history
  final_states:
    type: File[]
    outputSource: qsplit_instances/final_state
steps:
  prepare_dataset:
    run: clt/dataset_prepare.cwl
    in:
      configs: storage_configs
      dataset_jsonl: dataset
      max_instances: max_instances
      solutions_dir: solutions_store_dir
    out:
    - matrix_files
    - dataset_manifest
  qsplit_instances:
    run: instance.cwl
    in:
      split_configs: split_configs
      parallel_configs: parallel_configs
      iqm_configs: iqm_configs
      quantinuum_h2_configs: quantinuum_h2_configs
      quantinuum_h2e_configs: quantinuum_h2e_configs
      aggregate_configs: aggregate_configs
      storage_configs: storage_configs
      input_matrix: prepare_dataset/matrix_files
      cut_dim: cut_dim
      enable_sparse_check: enable_sparse_check
      enable_iqm: enable_iqm
      enable_quantinuum_h2: enable_quantinuum_h2
      enable_quantinuum_h2e: enable_quantinuum_h2e
      iqm_real_jobs: iqm_real_jobs
      quantinuum_h2_real_jobs: quantinuum_h2_real_jobs
      quantinuum_h2e_real_jobs: quantinuum_h2e_real_jobs
      solutions_dir: solutions_store_dir
      split_method: split_method
      aggregate_method: aggregate_method
      refinement_method: refinement_method
      refinement_loops: refinement_loops
      refinement_aggregate_method: refinement_aggregate_method
      refinement_configs: refinement_configs
      refinement_solver_configs: refinement_solver_configs
      refinement_backend: refinement_backend
      refinement_cut_dim: refinement_cut_dim
    out:
    - final_solutions
    - final_state
    - final_history
    scatter:
    - input_matrix
  collect_results:
    run: clt/collect_dataset_results.cwl
    in:
      configs: storage_configs
      dataset_manifest: prepare_dataset/dataset_manifest
      solutions_dir: solutions_store_dir
      solution_csv_list: qsplit_instances/final_solutions
    out:
    - results_dir
    - results_manifest
