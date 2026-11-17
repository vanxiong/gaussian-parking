#!/usr/bin/env bash
# Reproduce the worker without apt installs or changes to the baseline runtime.
set -euo pipefail
cd "$(dirname "$0")/.."
for executable in python3 gcc g++ git; do
    command -v "$executable" >/dev/null || { echo "Missing prerequisite: $executable"; exit 1; }
done
if [ ! -x .venv-linux/bin/python ]; then
    python3 -m venv .venv-linux
fi
source tools/linux_env.sh
python -m pip install --no-compile torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
if [ ! -x .toolchains/cuda/bin/nvcc ]; then
    python tools/install_cuda_toolchain.py
fi
python tools/restore_sources.py
bash tools/build_linux.sh
python tools/smoke_cuda.py
python tools/run_gaussianformer3d.py --build-only
python tools/check_environment.py --output reports/environment-linux.json
