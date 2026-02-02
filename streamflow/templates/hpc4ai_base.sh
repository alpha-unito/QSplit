#!/usr/bin/bash

#SBATCH --job-name=qsplit-base
#SBATCH --time=02:00:00

singularity exec /beegfs/home/fmedina/QSplit/streamflow/singularity/images/qsplit-base-aarch64.sif /bin/sh -s <<'STREAMFLOW_EOF'
{{streamflow_command}}
STREAMFLOW_EOF
