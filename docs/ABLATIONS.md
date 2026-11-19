# Reproducing ablations

Follow [installation](INSTALLATION.md) first: an external Isaac Sim installation,
WSL GPU/model environment, pinned sources/patches, frozen checkpoint and the
included car assets are required. A fresh clone does not need any archived
captures or reports. Run the Windows commands below from the repository root.

## Select the anchor policy explicitly

Every ablation runner requires one of:

- `--anchor-assignment random`: randomized inference anchor assignment, for
  comparison with the earlier experimental protocol.
- `--anchor-assignment deterministic`: canonical spatial ordering and a private
  fixed permutation. Training behavior and checkpoint tensors are unchanged.

The policy is applied when constructing the frozen model and recorded in the
experiment plan, comparison manifest, prediction timing records and Markdown
report. Deterministic assignment does not promise bitwise deterministic GPU
inference. Seeds, scoring masks and execution order remain controlled separately.
Resume requires the same policy, seed, poses and repeat count. Old manifests
without a policy are not silently interpreted by the ablation replay workers.

## Capture and compare from scratch

```powershell
# All four viewpoints, all appearance/building/resolution/history conditions.
.\evaluation\evaluate_multiview_ablations.bat --repeats 20 --anchor-assignment random

# Or run a single-pose study separately.
.\evaluation\evaluate_car_appearance_ablation.bat --repeats 20 --anchor-assignment random
.\evaluation\evaluate_building_ablation.bat --repeats 20 --anchor-assignment random
.\evaluation\evaluate_sensor_ablations.bat --repeats 20 --anchor-assignment random
```

Use `deterministic` instead of `random` to run a new study under that policy.
Do not mix policies when comparing members of an appearance/building/sensor
ablation. To compare the anchor policies themselves, use the tools below.

The multiview runner captures the original pose and every pose in
`configs/ablation_viewpoints.json` for each visual variant. It exports independent
geometry and derives the current-sweep replay from each reference capture.
`--poses` accepts a custom JSON list of unique `id`, `x`, `y`, `yaw_deg` records;
the original pose is added if absent. `--resume <output-directory>` reuses only
that experiment's saved capture state; no developer-specific report is needed.

The sensor runner creates a fresh 640-pixel baseline and native 960-pixel capture
when none are supplied. `--baseline-report` and `--high-resolution-report` can
reuse explicitly selected evaluation directories. Car/building runners likewise
offer paired report arguments; omit them to capture afresh.

## Compare both anchor policies on saved inputs

After a multiview run finishes, use its emitted `comparison.json` in WSL:

```bash
bash tools/evaluate_anchor_baseline.sh --comparison /path/to/comparison.json --repeats 20
bash tools/evaluate_anchor_assignment.sh --comparison /path/to/comparison.json
python tools/summarize_anchor_assignment.py /path/to/anchor-assignment-output
```

These commands deliberately evaluate **both** random and deterministic policies.
They read the supplied comparison's capture/ground-truth paths and verify input
hashes. Baseline comparison requires realistic reference captures at 640x360 with
eleven sweeps. The assignment diagnostic optionally accepts `--driving-run` for
first/middle/last capture stability checks and `--temporal-captures` for an ordered
set of captures at the same fixed sensor pose. Temporal inputs must be supplied
explicitly; a moving sequence is rejected by the fixed-pose check.

`bash tools/evaluate_anchor_assignment.sh --checks-only` runs assignment invariants
without saved captures, once the model environment and checkpoint are installed.
Use `--output` to choose a new output directory. Existing experiment outputs are
not overwritten. Summaries take their output directory as an argument.

## Keep outputs local

Default reports, predictions, figures and logs are under `reports/`; simulator
captures are under `simulation/data/`. Both are ignored by Git. Keep custom output
directories under `reports/` or `.cache/` as well. Commit scripts, configs, source
patches, tests and required assets, rather than generated experiment results.

The consolidated single-pose report accepts `--appearance`, `--building` and
`--sensor` output directories, plus optional `--input-probes` for an explicitly
selected exploratory probe file. It requires comparisons with the same recorded
anchor policy. Numerical results from earlier protocols remain historical.
