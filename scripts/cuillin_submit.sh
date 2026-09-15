#!/bin/bash
# Push the current commit and submit the corrector training on Cuillin. Run on the Mac:
#   bash scripts/cuillin_submit.sh
# The checkout on the cluster is a deploy target: it is reset to the pushed commit, so run
# scripts/cuillin_fetch.sh first if a previous job's outputs are still only there. Nothing
# here imports torch on the login node; the job script activates the venv on the worker. The
# job skips itself if runs/corrector.pt already exists in the checkout there.
set -e
HOST=${CUILLIN_HOST:-cuillin}
DIR=${CUILLIN_DIR:-/cephfs/mbandhu/magnetic-navigation}
BRANCH=$(git rev-parse --abbrev-ref HEAD)
git push origin "$BRANCH"
ssh "$HOST" "cd $DIR && git fetch origin && git checkout -q $BRANCH && git reset -q --hard origin/$BRANCH && mkdir -p logs && \
  LIST=scripts/sweeps/corrector.txt sbatch --gres=gpu:6000Ada:1 --array=1-1 scripts/slurm/cuillin_gpu.slurm && squeue -u mbandhu"
