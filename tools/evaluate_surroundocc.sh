#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/linux_env.sh"
export PYTHONPATH="$GF3D_ROOT/.eval-deps:${PYTHONPATH:-}"
exec "$GF3D_PYTHON" "$GF3D_ROOT/tools/evaluate_surroundocc.py" "$@"
