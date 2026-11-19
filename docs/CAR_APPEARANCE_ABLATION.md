# Parked-car appearance ablation

See the [consolidated report](../reports/ablation_study/20261118T055357_977940Z/REPORT.md)
for the completed appearance, building, native-resolution and LiDAR-history study.

The default scene remains `simple`. `realistic` replaces only the visual geometry
of all ten cars `/World/ParkedVehicles/Car0` through `Car9` with a pinned, imported
CC0 sedan (73 meshes, 16,300 triangles per car). The driving car, building,
lighting, and camera rig are unchanged. The original pilot used only cars 2, 4,
and 7; saved pilot manifests retain those IDs and remain distinguishable from
the full ten-car variant.

From `gaussian_parking`, launch either scene:

```powershell
.\drive_gaussianformer3d.bat --parked-car-appearance simple
.\drive_gaussianformer3d.bat --parked-car-appearance realistic
```

`drive_realistic_cars.bat` is a shortcut for the second command. Select the variant
at launch rather than switching midway through a timed capture.

Capture and evaluate at the same stationary pose:

```powershell
.\evaluation\evaluate_stationary.bat --parked-car-appearance simple
.\evaluation\evaluate_stationary.bat --parked-car-appearance realistic
```

Both accept the same `--start-offset`, `--start-y-offset`, and `--start-yaw-deg`.
Preserve each resulting report and manifest; `latest.json` points to the most
recent result. Replay with `evaluate_gaussianformer3d.bat --run <run-directory>
--frames 1`; the recorded appearance variant comes from the run's result.

### Repeated comparison at one pose

```powershell
.\evaluation\evaluate_car_appearance_ablation.bat --repeats 20 --anchor-assignment random
```

This captures **both variants afresh**, exports and scores their reference
geometry using the existing stationary evaluator, then runs 20 additional
inferences per variant with matching seeds (42 through 61 by default). Two
warm-up forwards per variant are excluded; simple-first and realistic-first
execution order alternate. Input checks require identical poses, calibration,
LiDAR arrays, depth guidance, exported reference objects, occupancy targets and
masks, and verify all ten detailed parked cars are present.

The original evaluator's `--repeats` option measures latency and retains only its
first prediction for accuracy. This dedicated comparison instead scores and
saves **every** prediction. It reports mean, sample SD, min/max, and paired mean
differences with conditional 95% t intervals. Those intervals describe inference
variation at this one pose, not uncertainty across parking lots or layouts.

Outputs are in `reports/appearance_ablation/<timestamp>/`: `COMPARISON.md`,
`comparison.json`, `runs.csv`, per-seed raw prediction arrays, a repeatability
figure, and a first-seed occupancy comparison. The plan, source manifests,
hashes, controls and execution logs are retained alongside them.
`validation.json` records independent recomputation of every replay's metrics
from saved predictions, summary arithmetic, pair ordering and CSV consistency.
`reports/appearance_ablation/latest.json` points to the completed comparison.
Existing report directories can be reused by supplying `--simple-report` and
`--realistic-report` together; the same input-equality checks still apply.

### Initial all-ten-car comparison — November 17, 2026

Fresh paired captures at ego world XY (-5, -2) m, yaw 0 degrees, with 20
inferences per variant (seeds 42–61). Values are mean ± sample SD; differences
use unrounded per-run values.

| Metric | Simple (%) | All 10 realistic (%) | Difference (pp) |
| --- | ---: | ---: | ---: |
| Occupied IoU | 17.57 ± 1.23 | 18.37 ± 0.96 | +0.80 |
| Precision | 25.95 ± 1.44 | 30.50 ± 1.31 | +4.54 |
| Recall | 35.19 ± 2.39 | 31.62 ± 1.85 | -3.57 |

The conditional 95% paired t interval for the mean IoU difference was
+0.20 to +1.41 percentage points. This supports a modest IoU increase for this
fixed capture, with a precision/recall tradeoff; it does not establish an
improvement across scenes. Pose, camera/LiDAR metadata, all 184,228 Nx5 LiDAR
samples, prepared depth/projections, exported reference objects, ground-truth
voxels and masks matched exactly. All six RGB inputs changed. All 40 saved
predictions were independently rescored and reconciled to the report and CSV.

[Full comparison and plots](../reports/appearance_ablation/20261118T051324_762878Z/COMPARISON.md)
and [machine-readable results](../reports/appearance_ablation/20261118T051324_762878Z/comparison.json).
UTC report-directory dates differ from the local evaluation date.

## What this comparison measures

This first experiment isolates **camera appearance**. Both variants keep original
simple-car collision shapes, simulated LiDAR geometry, and filled occupancy
references. The imported visual is fitted to the original car's complete bounding
dimensions, with the same center, yaw, ground clearance and paint color. Detailed
surfaces need not match the shared proxy exactly.

In the detailed variant, original parts receive USD `proxy` purpose while their
collision APIs stay unchanged. Explicit `occupancy:groundTruthProxy` attributes
tell the exporter to retain those shapes. The imported visual subtree has
`occupancy:excludeFromGroundTruth`; both lists are recorded by the exporter.
Unsupported nonconvex geometry elsewhere is still rejected; there is no
automatic bounding-box fallback.

Reports label the reference policy `shared_simple_car_proxy_v1`. Scores measure
agreement with shared proxies, **not exact detailed-mesh occupancy**. Changing
LiDAR/collision geometry or measuring exact mesh occupancy would be a separate
experiment requiring an appropriate ground-truth method.

Every run saves `appearance_ablation.json`; capture metadata, result JSON and the
evaluation manifest also record variant, car IDs and reference policy. Detailed
runs include source/converted asset hashes, dimensions and colors. Asset source,
CC0 provenance and conversion details are in `assets/parked_sedan/README.md`.
Simulation runs do not download assets.

## Verification and final study

Render an A/B preview and assert identical exported reference objects, evaluation
bounds, collision properties, all ten replacements and an unchanged driving car:

```powershell
.\runtime\python.bat tools\validate_car_appearance.py <simple-frame-directory>\scene.usda --output reports\car-appearance-validation --no-ros-env
```

This writes `validation.json` and `appearance-comparison.png`. The initial stage
pilot check retained 309 reference objects and excluded 219 detailed visual meshes
while retaining 57 original proxy parts. The ten-car version expects 730 detailed
visual meshes and 190 original proxy parts. The initial pilot capture also
matched the existing stationary baseline's 184,228 Nx5 LiDAR samples, ego pose
and camera metadata exactly.

For the final study, use several independent poses and repeated inference with
matching settings. Known random-anchor/runtime variation means a single IoU
difference is not evidence of improvement. Report whole-scene IoU, precision and
recall; add matching predefined regions around the parked cars as a
secondary comparison. Do not choose regions or thresholds based on improvement.
