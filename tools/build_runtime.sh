#!/usr/bin/env bash
# Snapshot installed dependencies; no downloads or package/version changes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/.runtime-linux"
if mountpoint -q "$ROOT/.runtime-linux/mount"; then
    echo 'Stop the model service and unmount the runtime before rebuilding.' >&2; exit 1
fi
mksquashfs "$ROOT/.venv-linux" "$ROOT/vendor/mmcv" "$ROOT/vendor/DFA3D" \
    "$ROOT/.runtime-linux/dependencies.squashfs.new" -noappend -comp lz4 -processors 4 -no-progress
mv "$ROOT/.runtime-linux/dependencies.squashfs.new" "$ROOT/.runtime-linux/dependencies.squashfs"
(cd "$ROOT" && sha256sum requirements-linux-lock.txt) > "$ROOT/.runtime-linux/dependencies.manifest"
