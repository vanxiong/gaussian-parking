#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/linux_env.sh"
"$GF3D_PYTHON" "$GF3D_ROOT/tools/compare_multiview_ablations.py" "$1"
