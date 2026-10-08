#!/bin/bash
# Grader validation on the real infrastructure:
#   oracle x N_ORACLE  -> every run must score 1.0 (and identically each time)
#   nop                -> must score 0.0
#   each cheat/<name>  -> must score <= cheat/<name>/expected_max
# Usage: scripts/validate.sh [oracle|nop|cheats|all]   (default: all)
source "$(dirname "$0")/common.sh"
WHAT="${1:-all}"
N_ORACLE="${N_ORACLE:-3}"
TS=$(stamp)
slug=$(basename "$TASK")
JOBS=()

run() { # job_name, extra harbor args...
  local job="$1"; shift
  harbor run -p "$TASK" -e "$ENV_TYPE" -o "$JOBS_DIR" --job-name "$job" -y "${ENV_FILE_ARGS[@]}" "$@" || true
  JOBS+=("$JOBS_DIR/$job")
}

if [[ $WHAT == all || $WHAT == oracle ]]; then
  run "$slug-oracle-$TS" -a oracle -k "$N_ORACLE" -n "$N_ORACLE"
fi
if [[ $WHAT == all || $WHAT == nop ]]; then
  run "$slug-nop-$TS" -a nop
fi
if [[ $WHAT == all || $WHAT == cheats ]]; then
  for cdir in "$TASK"/cheat/*/; do
    [ -f "$cdir/solve.sh" ] || continue
    cname=$(basename "$cdir")
    # Copy the task with cheat/<name> standing in as solution/, then run the oracle on it.
    build=".build/$slug-cheat-$cname"
    rm -rf "$build"; mkdir -p .build
    cp -R "$TASK" "$build"; rm -rf "$build/solution" "$build/cheat"
    cp -R "$cdir" "$build/solution"
    TASK_SAVE="$TASK"; TASK="$build"
    run "$slug-cheat-$cname-$TS" -a oracle
    TASK="$TASK_SAVE"
  done
fi

python3 scripts/summarize.py --check "$TASK" "${JOBS[@]}"
