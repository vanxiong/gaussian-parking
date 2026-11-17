# Upstream sources

`sources.json` records the exact upstream URLs and revisions used on this
workstation. The patches preserve tracked local changes, and `overlays/`
preserves additional source files. Upstream checkouts and build products are
excluded from the outer repository; their own licenses remain in the restored
checkouts. This repository does not relicense upstream code.

From `gaussian_parking`, restore sources with Python 3 and Git installed:

```sh
python tools/restore_sources.py
python tools/restore_sources.py --check
```

Restoration requires network access to the recorded upstream repositories.
Existing directories are verified without being overwritten. A partial or
divergent checkout causes an error and must be inspected before retrying.
The Linux bootstrap calls the same restoration helper.

This restores source only. Isaac Sim, Python/CUDA environments, frozen model
checkpoints and evaluation data must be installed or downloaded separately;
see the project README and maintenance/evaluation documentation. On this
workstation `runtime` points to a separate Isaac Sim installation. A fresh clone
creates its own link using `maintenance/link_isaac_sim.ps1`; see
[installation](../docs/INSTALLATION.md).
