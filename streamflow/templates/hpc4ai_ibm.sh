#!/usr/bin/bash

#SBATCH --job-name=qsplit-sibm
#SBATCH --time=02:00:00

singularity exec --bind /proc:/proc /beegfs/home/fmedina/QSplit/streamflow/singularity/images/qsplit-sibm-aarch64.sif /bin/sh -s <<'STREAMFLOW_EOF'
{{streamflow_command}}
STREAMFLOW_EOF
