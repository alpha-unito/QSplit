cwlVersion: v1.2
class: CommandLineTool
requirements:
  InlineJavascriptRequirement: {}
baseCommand:
- cli_refine
arguments:
- --phase
- update
inputs:
  input_state:
    type: File
    inputBinding:
      prefix: --input-state
  solved_list:
    type: File[]
    inputBinding:
      prefix: --solved-list
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
