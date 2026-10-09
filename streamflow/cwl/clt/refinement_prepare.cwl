cwlVersion: v1.2
class: CommandLineTool
requirements:
  InlineJavascriptRequirement: {}
baseCommand:
- cli_refine
arguments:
- --phase
- prepare
inputs:
  input_state:
    type: File
    inputBinding:
      prefix: --input-state
  running: boolean
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
  subproblems:
    type: File[]
    outputBinding:
      glob: subproblems/*.pkl
  solved_constant:
    type: File[]
    outputBinding:
      glob: solved_constant/*.pkl
