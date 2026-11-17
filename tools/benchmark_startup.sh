#!/usr/bin/env bash
# Fresh processes; deliberately do not flush OS/GPU caches or alter user settings.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for repetition in 1 2; do
    for mode in original fast; do
        (
            if [[ "$mode" == original ]]; then export GF3D_FAST_RUNTIME=0; else export GF3D_FAST_RUNTIME=1; fi
            source "$ROOT/tools/linux_env.sh"
            "$GF3D_PYTHON" "$ROOT/tools/benchmark_startup.py" "$ROOT/reports/performance/$mode-$repetition.json" \
                > "$ROOT/reports/performance/$mode-$repetition.log" 2>&1
        )
    done
done
