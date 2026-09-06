#!/bin/bash
# one GPU per run, one array task per line of the sweep file.
# submit with:  LIST=scripts/sweeps/main.txt sbatch --array=1-$(wc -l < scripts/sweeps/main.txt)%8 scripts/slurm/array.slurm
# fill in --account / --partition / --qos for your system (Cirrus: partition=gpu, qos=gpu, gres=gpu:1)
#SBATCH --job-name=magscale
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --account=CHANGEME
#SBATCH --partition=gpu
#SBATCH --qos=gpu

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs
# module load nvidia/cuda-12 python/3.11   # whatever your site needs
source .venv/bin/activate 2>/dev/null || true

LIST=${LIST:-scripts/sweeps/main.txt}
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
OUT=$(echo "$CMD" | sed -n 's/.*--out \([^ ]*\).*/\1/p')
if [ -f "$OUT" ]; then echo "already done: $OUT"; exit 0; fi
echo "$CMD"
eval "$CMD"
