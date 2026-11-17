#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/linux_env.sh"
cd "$GF3D_ROOT"
python tools/check_environment.py --output reports/environment-linux.json
