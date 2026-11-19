# Building-presence ablation

The building remains present by default. `--building absent` removes the complete
`/World/Building` subtree, including the main structure, windows and canopy.
It removes rendering, collisions, LiDAR returns and ground-truth occupancy together.
Other scene objects and the saved baseline scene are preserved.

With all ten parked cars using the detailed visual model, launch either variant:

```powershell
.\drive_realistic_cars.bat --building present
.\drive_realistic_cars.bat --building absent
```

One stationary capture/evaluation:

```powershell
.\evaluation\evaluate_stationary.bat --parked-car-appearance realistic --building absent
```

The paired repeated comparison captures both variants at the default settled
stationary pose and replays each 20 times:

```powershell
.\evaluation\evaluate_building_ablation.bat --repeats 20 --anchor-assignment random
```

The protocol matches the car-appearance experiment: seeds 42–61, matching within
each pair, alternating present-first and absent-first, two warm-up forwards per
variant, frozen checkpoint and unchanged primary occupancy policy. Exact control
checks require matching ego pose, camera calibration, realistic car metadata,
LiDAR configuration, evaluation masks and all nonbuilding geometry. Sensor images
and LiDAR returns deliberately change when the building disappears.

## Two score scopes

**Whole scene:** each variant uses its own ground truth on the same primary ROI.
The building-absent target contains fewer occupied cells. This reports performance
on two different scene-completion problems, not improved accuracy on identical
objects. The building's former occupied volume is evaluated as empty when absent.

**Common region:** exclude every voxel whose occupied ground truth disappears
with the building from both variants. The remaining evaluated cells have identical
ground truth. This is a separately reported diagnostic of prediction changes in
the unchanged environment. Its mask is derived from geometry before inference;
it does not replace the primary scores and is not chosen based on improvement.

The realistic cars retain the same original simple-car collision and occupancy
proxies in both variants. This experiment changes building presence rather than
changing the car reference policy.

Each capture records `building_ablation.json` and embeds the presence/removal scope
in capture metadata, result JSON, the evaluation manifest and report. Detailed
results are in `reports/building_ablation/<timestamp>/`: `COMPARISON.md`,
`comparison.json`, `runs.csv`, `controls.json`, geometry-derived comparison masks,
every raw prediction, repeatability and occupancy figures, the predeclared plan
and execution logs. `validation.json` records independent recomputation of all
whole-scene and common-region metrics from saved predictions.

Existing paired reports can be replayed using `--present-report` and
`--absent-report` together. Capture/configuration checks still apply. Repeated
inference measures model/runtime variation at one pose; it is not a sample of
independent parking-lot layouts or evidence of generalization.

## Initial measured comparison — November 17, 2026

Twenty paired inference runs per variant, all ten parked cars realistic. Values
are mean ± sample SD, in percent.

| Whole scene | Building present | Building absent | Difference (pp) |
| --- | ---: | ---: | ---: |
| Occupied IoU | 18.12 ± 1.36 | 14.02 ± 0.54 | -4.11 |
| Precision | 29.93 ± 1.80 | 15.31 ± 0.57 | -14.62 |
| Recall | 31.45 ± 2.26 | 62.30 ± 1.80 | +30.85 |

Whole-scene occupied targets changed from 6,777 to 1,612 cells. For the
identical-target common region, excluding the 5,165 removed occupied cells:

| Common region | Building present | Building absent | Difference (pp) |
| --- | ---: | ---: | ---: |
| Occupied IoU | 16.18 ± 0.55 | 17.64 ± 0.66 | +1.46 |
| Precision | 17.64 ± 0.64 | 19.75 ± 0.70 | +2.10 |
| Recall | 66.22 ± 1.50 | 62.30 ± 1.80 | -3.92 |

The common-region IoU difference had a conditional 95% paired t interval of
+1.11 to +1.80 percentage points. The whole-scene recall increase does not
indicate better recall on the remaining objects: their recall declined slightly.
All 40 predictions were independently rescored for both scope definitions.

[Full comparison](../reports/building_ablation/20261118T052211_821456Z/COMPARISON.md)
and [machine-readable results](../reports/building_ablation/20261118T052211_821456Z/comparison.json).

## Building-free semantic and LiDAR diagnosis

Analyze all saved building-free replays without new inference:

```bash
source tools/linux_env.sh
"$GF3D_PYTHON" tools/diagnose_building_free.py reports/building_ablation/20261118T052211_821456Z/comparison.json
```

The diagnosis is in that comparison's `extra_occupancy/` directory. It includes
per-seed/per-class false positives, height bands, exact current-return ray/voxel
crossings, same-cell hit support from both current and all saved sweeps,
nearest-above-ground-return distances, and spatial/class figures. Rays stop
0.5 m before measured hits. Missing/no-return beams are not assumed free, and
a ray crossing only observes part of a cell. False-positive status comes from
independent reference geometry. Six analytical ray fixtures cover margins,
negative directions, corner crossings, grid exits, short rays and outside hits.

Across 20 replays, false-positive labels were 39.16% driveable surface, 21.49% car,
18.81% manmade, 8.77% sidewalk and 6.54% traffic cone. In the former building
volume they were 58.02% driveable surface, 24.01% manmade, 11.85% sidewalk and
6.13% car. No false-positive voxel contained a hit from any saved sweep.
40.35% of all false positives intersected current measured free-ray segments;
this fell to 17.28% inside the former building volume. The remaining evidence
is unresolved, not confirmed occupied. All current returns aligned to exported
geometry within the 5 cm convex-surface-plane diagnostic tolerance.

[Full class/LiDAR diagnosis](../reports/building_ablation/20261118T052211_821456Z/extra_occupancy/DIAGNOSIS.md).
