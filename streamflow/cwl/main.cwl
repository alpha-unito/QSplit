cwlVersion: v1.2
class: Workflow

requirements:
  - class: ScatterFeatureRequirement
  - class: StepInputExpressionRequirement
  - class: InlineJavascriptRequirement

inputs:
  input_matrix: File
  backend_cut_dims: string

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
      backend_cut_dims: backend_cut_dims
    out: [sub_qubos, full_qubo, tree_meta]

  parallelize:
    run: clt/scatter.cwl
    in:
      input_qubo: split/sub_qubos
      backend:
        valueFrom: |
          ${
            var p = inputs.input_qubo.path;
            var parts = p.split("/");
            return parts[parts.length - 2];
          }
    out: [solved_qubo]
    scatter: [input_qubo]

  aggregate:
    run: clt/aggregate.cwl
    in:
      input_qubo: split/full_qubo
      tree_file: split/tree_meta
      solved_list: parallelize/solved_qubo
    out: [aggregate_solutions]