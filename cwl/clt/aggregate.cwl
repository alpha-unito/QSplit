cwlVersion: v1.2
class: CommandLineTool

requirements:
  - class: InlineJavascriptRequirement

baseCommand: ["python", "-m", "qsplit.cli_aggregate"]

inputs:
  input_qubo:
    type: File
    inputBinding:
      prefix: --input-qubo

  solved_list:
    type: File[]
    inputBinding:
      prefix: --solved-list
      valueFrom: $(self.map(function (f) { return f.path; }).join(","))

  output_qubo_name:
    type: string
    inputBinding:
      prefix: --output-qubo

outputs:
  aggregated_csv:
    type: File
    outputBinding:
      glob: $(inputs.output_qubo_name.replace(".pkl", ".csv"))