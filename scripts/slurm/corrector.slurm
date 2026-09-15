#!/bin/bash
# Cuillin CPU array for the simulator's corrector (magsim/train_corrector.py), one task per
# line of scripts/sweeps/corrector.txt. About two hours per line on eight cores.
#   sbatch --array=1-$(wc -l < scripts/sweeps/corrector.txt) scripts/slurm/corrector.slurm
#SBATCH --job-name=magsim-corrector
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/corrector_%A_%a.out

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results runs
source .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
LIST=${LIST:-scripts/sweeps/corrector.txt}
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
echo "$CMD"
eval "$CMD"
