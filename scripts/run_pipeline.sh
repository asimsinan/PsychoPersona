#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-configs/runs/paper_validation_baseline.yaml}"
RUN_DIR="${2:-}"

if [[ -n "$RUN_DIR" ]]; then
  run_dir="$(
    cd "$PROJECT_ROOT" &&
    uv run saykopersona-experiment --config "$CONFIG_PATH" --run-dir "$RUN_DIR" --resume --step all >&2 &&
    printf '%s\n' "$RUN_DIR"
  )"
else
  run_dir="$(
    cd "$PROJECT_ROOT" &&
    uv run saykopersona-experiment --config "$CONFIG_PATH" --step all >&2 &&
    # sed reads the whole listing; `head -1` closes the pipe early, and under
    # `pipefail` that turned a finished run into a failed script now and then.
    ls -1dt results/runs/experiment/*/ | sed -n 1p
  )"
fi

printf '%s\n' "$run_dir"
