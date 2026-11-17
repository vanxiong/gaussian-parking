#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/linux_env.sh"
if [[ ! -d "$GF3D_ROOT/.eval-deps/pyquaternion-0.9.9.dist-info" ]]; then
    "$GF3D_PYTHON" -m pip install --disable-pip-version-check --no-deps --target "$GF3D_ROOT/.eval-deps" -r "$GF3D_ROOT/requirements-evaluation-lock.txt"
fi
echo 'Evaluation dependency ready. Dataset downloads and account access are separate; see docs/SURROUNDOCC_EVALUATION.md.'
