#!/usr/bin/bash

#SBATCH --job-name=qsplit-gpu
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1


REPO_DIR=/beegfs/home/fmedina/QSplit

source "$REPO_DIR/.venvs/qsplit-epito/bin/activate"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR"

{{streamflow_command}}
