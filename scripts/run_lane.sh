#!/usr/bin/env bash
# Run one lane of a generation batch. Reads config paths, one per line, from
# "$SCRATCH/$LANE.txt" and writes per-run logs into "$SCRATCH/logs".
#
# Retries a 402 three times with a five-minute backoff, because a top-up is an external
# event worth waiting for, and fails fast on anything else so a real error is not
# masked by repeated attempts.
#
# Usage: run_lane.sh <scratch dir> <lane name>
set -u
SCRATCH="$1"; LANE="$2"
cd /workspace/PsychoPersona
set -a; . ./.env; set +a
mkdir -p "$SCRATCH/logs"
while read -r cfg; do
  [ -z "$cfg" ] && continue
  name=$(basename "$cfg" .yaml)
  for attempt in 1 2 3; do
    if bash scripts/run_pipeline.sh "$cfg" > "$SCRATCH/logs/${name}.log" 2>&1; then
      echo "[$(date +%H:%M:%S)] OK    $name"; break
    fi
    if grep -q "402 Payment Required" "$SCRATCH/logs/${name}.log"; then
      echo "[$(date +%H:%M:%S)] 402   $name waiting 300s"; sleep 300
    else
      echo "[$(date +%H:%M:%S)] FAIL  $name"; break
    fi
    [ "$attempt" = 3 ] && echo "[$(date +%H:%M:%S)] FAIL  $name (402 x3)"
  done
done < "$SCRATCH/$LANE.txt"
echo "[$(date +%H:%M:%S)] DONE $LANE"
