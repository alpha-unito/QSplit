#!/usr/bin/bash

#SBATCH --job-name=qsplit-cpu
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

REPO_DIR=/beegfs/home/fmedina/QSplit

source "$REPO_DIR/.venvs/qsplit-cpu/bin/activate"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR"

{{streamflow_command}}
