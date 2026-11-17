#!/usr/bin/env bash
# Run as WSL root. Only mounts the project's dependency image, read-only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="$ROOT/.runtime-linux/dependencies.squashfs"
TARGET="$ROOT/.runtime-linux/mount"
test -f "$IMAGE"
unsquashfs -s "$IMAGE" >/dev/null
if [[ -f "$ROOT/.runtime-linux/dependencies.manifest" ]]; then
    # Older snapshots recorded an absolute filename. Verify the lock at this
    # checkout's root so moving/renaming the project does not invalidate it.
    read -r expected_hash _ < "$ROOT/.runtime-linux/dependencies.manifest"
    actual_hash="$(sha256sum "$ROOT/requirements-linux-lock.txt")"
    if [[ ! "$expected_hash" =~ ^[0-9a-f]{64}$ ]] || [[ "$expected_hash" != "${actual_hash%% *}" ]]; then
        echo 'Dependency lock changed. Rebuild the runtime snapshot.' >&2; exit 1;
    fi
fi
mkdir -p "$TARGET"
if mountpoint -q "$TARGET"; then
    test "$(findmnt -n -o FSTYPE --target "$TARGET")" = squashfs
else
    mount -t squashfs -o loop,ro,nosuid,nodev "$IMAGE" "$TARGET"
fi
