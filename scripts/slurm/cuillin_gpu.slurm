#!/bin/bash
# Cuillin (ROE) GPU array: one run per task, one GPU per task.
#   LIST=scripts/sweeps/colab_main.txt sbatch --gres=gpu:6000Ada:1 --array=1-68%4 scripts/slurm/cuillin_gpu.slurm
#   LIST=scripts/sweeps/colab_smallD.txt sbatch --gres=gpu:2080Ti:1 --array=1-32%4 scripts/slurm/cuillin_gpu.slurm
#SBATCH --job-name=magscale
#SBATCH --gres=gpu:1
# the card type is chosen on the command line: sbatch --gres=gpu:6000Ada:1 ... overrides this line
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%A_%a.out

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs
source .venv/bin/activate

LIST=${LIST:-scripts/sweeps/colab_main.txt}
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
OUT=$(echo "$CMD" | sed -n 's/.*--out \([^ ]*\).*/\1/p')
if [ -f "$OUT" ]; then echo "already done: $OUT"; exit 0; fi
echo "$CMD"
nvidia-smi --query-gpu=name --format=csv,noheader
eval "$CMD"
