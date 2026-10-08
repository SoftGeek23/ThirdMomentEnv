# Sourced by the other scripts.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TASK="${TASK:-tasks/qwen3-w8a8-recipe}"
ENV_TYPE="${ENV_TYPE:-modal}"
JOBS_DIR="${JOBS_DIR:-jobs}"
ENV_FILE_ARGS=()
[ -f .env ] && ENV_FILE_ARGS=(--env-file .env)
stamp() { date +%Y%m%d-%H%M%S; }
