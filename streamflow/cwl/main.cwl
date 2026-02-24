cwlVersion: v1.2
class: Workflow

requirements:
  - class: ScatterFeatureRequirement
  - class: MultipleInputFeatureRequirement
  - class: SubworkflowFeatureRequirement

inputs:
  dataset: File
  max_instances:
    type:
      - "null"
      - int
    default: null
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

steps:
  prepare_dataset:
    run: clt/dataset_prepare.cwl
    in:
      dataset_jsonl: dataset
      max_instances: max_instances
    out: [matrix_files, dataset_manifest]

  qsplit_instances:
    run: instance.cwl
    in:
      input_matrix: prepare_dataset/matrix_files
      cut_dim: cut_dim
      enable_sparse_check: enable_sparse_check
      enable_iqm: enable_iqm
      enable_quantinuum_h2: enable_quantinuum_h2
      enable_quantinuum_h2e: enable_quantinuum_h2e
      iqm_real_jobs: iqm_real_jobs
      quantinuum_h2_real_jobs: quantinuum_h2_real_jobs
      quantinuum_h2e_real_jobs: quantinuum_h2e_real_jobs
    out: [final_solutions]
    scatter: [input_matrix]

  collect_results:
    run: clt/collect_dataset_results.cwl
    in:
      dataset_manifest: prepare_dataset/dataset_manifest
      solution_csv_list: qsplit_instances/final_solutions
    out: [results_dir, results_manifest]
