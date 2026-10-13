#!/bin/sh
# Replace with the worker's QSplit environment. This is an SSH service template.
export PATH="/absolute/path/to/remote/QSplit/.venv/bin:$PATH"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
{{streamflow_command}}
