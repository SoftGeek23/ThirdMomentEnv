#!/bin/bash
# The 5 graded agent rollouts, using exactly the settings in configs/rollouts.yaml.
# Usage: scripts/rollouts.sh [job-name]
source "$(dirname "$0")/common.sh"
[ -n "${OPENROUTER_API_KEY:-}" ] || grep -q '^OPENROUTER_API_KEY=.\+' .env 2>/dev/null \
  || { echo "OPENROUTER_API_KEY not set (export it or put it in .env)"; exit 1; }
JOB="${1:-$(basename "$TASK")-rollouts-$(stamp)}"
harbor run -c configs/rollouts.yaml -o "$JOBS_DIR" --job-name "$JOB" -y "${ENV_FILE_ARGS[@]}" || true
cp configs/rollouts.yaml "$JOBS_DIR/$JOB/rollouts.config.yaml"
python3 scripts/summarize.py --markdown "report/$JOB.md" "$JOBS_DIR/$JOB"
