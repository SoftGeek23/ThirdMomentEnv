#!/bin/bash
# Static checks: the Terminal-Bench v4.0.0 CI checks (reference/tb-checks) plus
# Harbor's task-config parse.
source "$(dirname "$0")/common.sh"
# Checks that only make sense inside the terminal-bench repo itself.
# check-task-package-name: requires [task] name = "terminal-bench/<slug>".
# check-gpu-types: TB list has "A10"; Modal (our backend) names the card "A10G".
SKIP=" check-task-package-name check-gpu-types "
fail=0
for c in reference/tb-checks/check-*.sh; do
  name=$(basename "$c" .sh)
  [[ $SKIP == *" $name "* ]] && { echo "SKIP $name"; continue; }
  if out=$(bash "$c" "$TASK" 2>&1); then
    echo "PASS $name"
  else
    echo "FAIL $name"; echo "$out" | sed 's/^/     /' | tail -8; fail=1
  fi
done
exit $fail
