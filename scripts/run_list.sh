#!/bin/bash
# run every line of a sweep file in order, skipping runs whose json already exists
# usage: bash scripts/run_list.sh scripts/sweeps/main.txt
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  out=$(echo "$cmd" | sed -n 's/.*--out \([^ ]*\).*/\1/p')
  if [ -f "$out" ]; then echo "skip $out"; continue; fi
  echo "$cmd"
  eval "$cmd" || echo "FAILED: $cmd" >> scripts/sweeps/failed.txt
done < "$1"
