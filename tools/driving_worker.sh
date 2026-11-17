#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/linux_env.sh"
cd "$GF3D_ROOT"
export PYTHONUNBUFFERED=1
exec python -u tools/driving_worker.py
