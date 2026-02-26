cwlVersion: v1.2
class: ExpressionTool

requirements:
  - class: InlineJavascriptRequirement

inputs:
  parallel_solved:
    type: File[]
    default: []
  iqm_solved:
    type: File[]
    default: []
  quantinuum_h2_solved:
    type: File[]
    default: []
  quantinuum_h2e_solved:
    type: File[]
    default: []
  split_solved:
    type: File[]
    default: []

outputs:
  solved_list: File[]

expression: |
  ${
    const merged = [];
    const append = (items) => {
      if (!Array.isArray(items)) return;
      for (const item of items) {
        if (item) merged.push(item);
      }
    };
    append(inputs.parallel_solved);
    append(inputs.iqm_solved);
    append(inputs.quantinuum_h2_solved);
    append(inputs.quantinuum_h2e_solved);
    append(inputs.split_solved);
    return { solved_list: merged };
  }
