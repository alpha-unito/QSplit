#!/usr/bin/bash

#SBATCH --job-name=qsplit-sibm
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# source /beegfs/home/fmedina/spack/share/spack/setup-env.sh
# spack load apptainer
singularity exec --nv --bind /proc:/proc /beegfs/home/fmedina/QSplit/streamflow/singularity/images/qsplit-sibm-gpu.sif /bin/sh -s <<'STREAMFLOW_EOF'
{{streamflow_command}}
STREAMFLOW_EOF
