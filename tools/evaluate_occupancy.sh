#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/linux_env.sh"
"$GF3D_PYTHON" "$GF3D_ROOT/tools/benchmark_occupancy.py" "$1"
"$GF3D_PYTHON" "$GF3D_ROOT/tools/evaluate_occupancy.py" "$1"
"$GF3D_PYTHON" "$GF3D_ROOT/tools/report_occupancy_evaluation.py" "$1"
