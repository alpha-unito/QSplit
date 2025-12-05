cwlVersion: v1.2
class: Workflow

requirements:
  - class: ScatterFeatureRequirement

inputs:
  input_matrix: File
  cut_dim: int
  backends: string
  backend_cut_dims: string

outputs:
  final_solution:
    type: File
    outputSource: aggregate/aggregated_csv

steps:
  split:
    run: clt/split.cwl
    in:
      input_qubo: input_matrix
      cut_dim: cut_dim
      backends: backends
      adaptive: { default: true }
      backend_cut_dims: backend_cut_dims
    out: [sub_qubos, full_qubo, tree_meta, sub_backends]

  dwave_solve:
    run: clt/dwave_solve.cwl
    in:
      input_qubo: split/sub_qubos
      backend: split/sub_backends
    out: [solved_qubo]
    scatter: [input_qubo, backend]
    scatterMethod: dotproduct

  aggregate:
    run: clt/aggregate.cwl
    in:
      input_qubo: split/full_qubo
      solved_list: dwave_solve/solved_qubo
      tree_file: split/tree_meta
    out: [aggregated_csv]