#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/linux_env.sh"
export MAX_JOBS=4
python "$GF3D_ROOT/tools/merge_cuda_headers.py"
cd "$GF3D_ROOT/vendor/DFA3D/DFA3D"
python setup.py build_ext --inplace
cd "$GF3D_ROOT/GaussianFormer3D/model/head/localagg"
python setup.py build_ext --inplace
cd "$GF3D_ROOT/GaussianFormer3D/model/encoder/gaussian_encoder/ops"
python setup.py build_ext --inplace
