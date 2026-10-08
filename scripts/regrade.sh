#!/bin/bash
# Re-grade finished trials' saved submissions with the CURRENT grader (tests/), without
# re-running the agents. Each trial's artifacts/app/submission/ becomes the "solution" of a
# throwaway task copy, which Harbor's oracle agent copies into /app/submission before the
# (current) verifier runs on a fresh GPU.
#
#   scripts/regrade.sh jobs/<rollout-job> [jobs/<rollout-job> ...]
#   -> one new job per trial: jobs/regrade-<trial>-<timestamp>; summary at the end.
source "$(dirname "$0")/common.sh"
TS=$(stamp)
JOBS=()
pids=()
for job in "$@"; do
  for trial in "$job"/*/; do
    sub="$trial/artifacts/app/submission"
    [ -f "$sub/recipe.py" ] || { echo "skip $(basename "$trial"): no saved submission"; continue; }
    name=$(basename "$trial")
    build=".build/regrade-$name"
    rm -rf "$build"; mkdir -p .build
    cp -R "$TASK" "$build"; rm -rf "$build/solution" "$build/cheat"
    mkdir -p "$build/solution/submission"
    rsync -a --exclude __pycache__ "$sub/" "$build/solution/submission/"
    cat > "$build/solution/solve.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
mkdir -p /app/submission
cp -R /solution/submission/. /app/submission/
EOF
    chmod +x "$build/solution/solve.sh"
    jn="regrade-$name-$TS"
    harbor run -p "$build" -e "$ENV_TYPE" -o "$JOBS_DIR" --job-name "$jn" -y "${ENV_FILE_ARGS[@]}" -a oracle \
      > "$JOBS_DIR/.$jn.log" 2>&1 &
    pids+=($!)
    JOBS+=("$JOBS_DIR/$jn")
    echo "regrading $name -> $jn"
  done
done
wait "${pids[@]}"
python3 scripts/summarize.py "${JOBS[@]}"
