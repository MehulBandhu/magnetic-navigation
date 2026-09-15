#!/bin/bash
# Fetch the trained corrector from Cuillin and render the held-out table. Run on the Mac:
#   bash scripts/cuillin_fetch.sh
set -e
HOST=${CUILLIN_HOST:-cuillin}
DIR=${CUILLIN_DIR:-/cephfs/mbandhu/magnetic-navigation}
mkdir -p results runs
for f in results/corrector.json runs/corrector.json runs/corrector.pt; do
  rsync -av "$HOST:$DIR/$f" "$f"
done
source .venv/bin/activate
python scripts/render_tables.py > /dev/null
sed -n '/^## Simulator: residual after each estimator/,/^The baselines/p' docs/results_tables.md
