#!/bin/bash
# Cuillin CPU array for the linear-learner reference curves (no GPU needed).
#   LIST=scripts/sweeps/linear.txt sbatch --array=1-$(wc -l < scripts/sweeps/linear.txt)%10 scripts/slurm/cuillin_cpu.slurm
#SBATCH --job-name=magscale-lin
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/lin_%A_%a.out

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results
source .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
LIST=${LIST:-scripts/sweeps/linear.txt}
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
echo "$CMD"
eval "$CMD"
