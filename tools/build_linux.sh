#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/linux_env.sh"
cd "$GF3D_ROOT"
python tools/patch_upstream.py
python -m pip install -r requirements-linux.txt
python tools/merge_cuda_headers.py
cd "$GF3D_ROOT/vendor/mmcv"
MMCV_WITH_OPS=1 FORCE_CUDA=1 python setup.py build_ext --inplace
MMCV_WITH_OPS=1 python -m pip install --no-build-isolation --no-deps -e .
cd "$GF3D_ROOT/vendor/DFA3D/DFA3D"
python setup.py build_ext --inplace
python -m pip install --no-build-isolation --no-deps -e .
cd "$GF3D_ROOT/GaussianFormer3D/model/head/localagg"
python setup.py build_ext --inplace
cd "$GF3D_ROOT/GaussianFormer3D/model/encoder/gaussian_encoder/ops"
python setup.py build_ext --inplace
cd "$GF3D_ROOT"
python -m pip freeze > requirements-linux-lock.txt
