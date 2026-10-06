cwlVersion: v1.2
class: CommandLineTool
requirements:
  InlineJavascriptRequirement: {}
baseCommand:
- cli_refine
arguments:
- --phase
- initialize
inputs:
  input_state:
    type: File
    inputBinding:
      prefix: --input-state
  configs:
    type:
      type: array
      items: File
      inputBinding:
        prefix: --config
    default: []
    inputBinding: {}
  method:
    type: string
    default: none
    inputBinding:
      prefix: --method
  loops:
    type: int
    default: 0
    inputBinding:
      prefix: --loops
  cut_dim:
    type: int
    default: 16
    inputBinding:
      prefix: --cut-dim
  aggregate_method:
    type: string
    default: linear
    inputBinding:
      prefix: --aggregate-method
outputs:
  state:
    type: File
    outputBinding:
      glob: state.pkl
  continue_refinement:
    type: boolean
    outputBinding:
      glob: control.json
      loadContents: true
      outputEval: $(JSON.parse(self[0].contents).continue)
  solutions:
    type: File
    outputBinding:
      glob: solutions.csv
  history:
    type: File
    outputBinding:
      glob: refinement_history.json
