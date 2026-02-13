cwlVersion: v1.2
class: Workflow

requirements:
  - class: ScatterFeatureRequirement
  - class: MultipleInputFeatureRequirement

inputs:
  input_matrix: File
  cut_dim: int

outputs:
  final_solutions:
    type: File
    outputSource: aggregate/aggregate_solutions

steps:
  split:
    run: clt/split.cwl
    in:
      input_qubo: input_matrix
      adaptive: { default: true }
      cut_dim: cut_dim
    out: [sub_qubos, solved_qubos, full_qubo, tree_meta]

  parallelize:
    run: clt/scatter.cwl
    in:
      input_qubo: split/sub_qubos
    out: [solved_qubo]
    scatter: [input_qubo]

  aggregate:
    run: clt/aggregate.cwl
    in:
      input_qubo: split/full_qubo
      tree_file: split/tree_meta
      solved_list:
        source: [parallelize/solved_qubo, split/solved_qubos]
        linkMerge: merge_flattened
    out: [aggregate_solutions]
