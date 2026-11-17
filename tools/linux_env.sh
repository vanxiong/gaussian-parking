#!/usr/bin/env bash
# Source from a shell; all mutable model/build caches stay in the project.
GF3D_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GF3D_VENV="$GF3D_ROOT/.venv-linux"
# Editable-install metadata may contain the checkout's old absolute path.
# Resolve vendor sources from this checkout when the optional mount is absent.
export PYTHONPATH="$GF3D_ROOT/vendor/mmcv:$GF3D_ROOT/vendor/DFA3D/DFA3D:${PYTHONPATH:-}"
if [[ "${GF3D_FAST_RUNTIME:-1}" != 0 ]] && mountpoint -q "$GF3D_ROOT/.runtime-linux/mount"; then
    GF3D_VENV="$GF3D_ROOT/.runtime-linux/mount/.venv-linux"
    export PYTHONPATH="$GF3D_ROOT/.runtime-linux/mount/mmcv:$GF3D_ROOT/.runtime-linux/mount/DFA3D/DFA3D:${PYTHONPATH:-}"
fi
export GF3D_PYTHON="$GF3D_VENV/bin/python"
export PATH="$GF3D_VENV/bin:$GF3D_ROOT/.toolchains/cuda/bin:$PATH"
export CUDA_HOME="$GF3D_ROOT/.toolchains/cuda"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$GF3D_VENV/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export PIP_CACHE_DIR="$GF3D_ROOT/.cache/linux-pip"
export XDG_CACHE_HOME="$GF3D_ROOT/.cache"
export TORCH_EXTENSIONS_DIR="$GF3D_ROOT/.cache/torch-extensions"
export TMPDIR="$GF3D_ROOT/.cache/tmp"
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=8
export MPLCONFIGDIR="$GF3D_ROOT/.cache/matplotlib"
export NUMBA_CACHE_DIR="$GF3D_ROOT/.cache/numba"
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$MPLCONFIGDIR"
