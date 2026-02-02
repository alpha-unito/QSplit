cwlVersion: v1.2
class: Workflow

requirements:
  - class: ScatterFeatureRequirement
  - class: StepInputExpressionRequirement
  - class: InlineJavascriptRequirement
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
        valueFrom: |-
          ${
            var out = [];
            if (self != null) {
              if (self.length && (self[0] instanceof Array || self[1] instanceof Array)) {
                for (var i = 0; i < self.length; i++) {
                  var arr = self[i] || [];
                  for (var j = 0; j < arr.length; j++) {
                    if (arr[j] != null) out.push(arr[j]);
                  }
                }
              } else {
                for (var k = 0; k < self.length; k++) {
                  if (self[k] != null) out.push(self[k]);
                }
              }
            }
            return out;
          }
    out: [aggregate_solutions]
