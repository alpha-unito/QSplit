cwlVersion: v1.2
class: Workflow

inputs:
  input_matrix: File
  cut_dim: int

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
      approach: { default: "dr" }
      adaptive: { default: true }
      out_dir: { default: "subproblems" }
      tree_file: { default: "tree.json" }
    out: [sub_qubos, full_qubo, tree_meta]

  dwave_solve:
    run: clt/dwave_solve.cwl
    in:
      input_qubo: split/sub_qubos
      output_qubo_name: { default: "solved.pkl" }
    out: [solved_qubo]
    scatter: input_qubo
    scatterMethod: flat_crossproduct

  aggregate:
    run: clt/aggregate.cwl
    in:
      input_qubo: split/full_qubo
      solved_list: dwave_solve/solved_qubo
      tree_file: split/tree_meta
      output_qubo_name: { default: "aggregated.pkl" }
    out: [aggregated_csv]
