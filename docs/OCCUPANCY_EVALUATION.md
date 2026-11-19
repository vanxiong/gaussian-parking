# GaussianFormer3D versus Isaac Sim geometry

The project now has a measured geometric occupancy baseline, separate from driving success. All work and data stay in `gaussian_parking`. No ROS process is required and weights remain frozen.

## Run it

The stationary launcher accepts `--parked-car-appearance simple` (default) or
`--parked-car-appearance realistic` for an all-10-parked-car visual ablation. Both use
identical original car collision and occupancy reference shapes. The detailed
variant is explicitly labeled as a shared-proxy appearance test; see
[car appearance ablation](CAR_APPEARANCE_ABLATION.md) for commands and scope.

For the all-realistic-car building-present/absent comparison, use
`evaluation\evaluate_building_ablation.bat --repeats 20 --anchor-assignment random`. See
[building ablation](BUILDING_ABLATION.md) for per-scene versus identical-target
score scopes and the `--building present|absent` toggle.

For a fresh stationary parking-lot evaluation, run from `gaussian_parking`:

```powershell
.\evaluation\evaluate_stationary.bat
```

The car settles at the default parking-lot starting position, then physics pauses.
One synchronized six-camera/LiDAR capture and independent USD snapshot are saved;
no driving or path search is performed. The existing evaluator replays the capture
and reports occupied IoU, precision, recall, and `occupancy-comparison.png` (a
horizontal slice near world height 1 m, not a projection of the full 3D score).
The report records the fixed world pose. `--start-offset`, `--start-y-offset`
(meters relative to the default start), and `--start-yaw-deg` select another pose.
Capture logs are in `reports/stationary-capture.log`; evaluation outputs and
`reports/evaluation/latest.json` use the same format as the existing evaluator.
Offline timing repeats do not add independent accuracy observations. The primary
score retains the existing full-geometry, occlusion-inclusive protocol below.

From `gaussian_parking`, capture paired camera/LiDAR inputs and exact per-frame USD snapshots:

```powershell
.\drive_gaussianformer3d.bat --headless --evaluation-snapshots --max-sim-seconds 120
.\evaluation\evaluate_gaussianformer3d.bat
```

The second command chooses the latest run with paired snapshots. For a specific run:

```powershell
.\evaluation\evaluate_gaussianformer3d.bat --run simulation/data/driving_runs/20261116T001235_529229Z --frames 5 --warmup 2 --repeats 3
```

Reports go to `reports/evaluation/<timestamp>/`; `reports/evaluation/latest.json` points to the latest completed report. Each report includes `REPORT.md`, accuracy and latency JSON, per-frame voxel ground truth and masks, fresh model predictions, scene hashes, geometry exports and a PNG comparison. Existing capture data is preserved. The dependency snapshot is mounted automatically. The USD exporter briefly starts Isaac's runtime; it closes before offline GPU timing begins.

## What is ground truth?

Ground truth comes from the actual rendered USD geometry, not LiDAR hit points, a planner costmap, or model labels. `--evaluation-snapshots` exports a scene at each synchronized camera/LiDAR snapshot. The input manifest records its timestamp and export overhead. The evaluation checks the exported ego pose against the capture pose and verifies that timestamps match.

The exporter removes the ego car, debug overlays, invisible prims and guide/proxy geometry. It handles cubes, cylinders, cones, spheres and closed convex meshes. Mesh closure, planarity and convexity are checked; unsupported or nonconvex visible geometry causes an error rather than silently becoming a bounding box. Curved primitives use 32 angular segments; spheres use 16 latitude bands. This is an approximation of curved surfaces at much finer spacing than the voxel size, with possible boundary-cell differences.

Each transformed shape is voxelized as a **filled solid** using interior tests and triangle/voxel intersection. This retains thin objects that voxel-center-only sampling could miss. The union is mapped into the capture's LiDAR frame on the model's exact **200 x 200 x 16 XYZ grid**, origin **[-50,-50,-5] m**, **0.5 m** cells. The model receives only RGB, calibration and LiDAR; the ground-truth exporter and scorer are separate processes.

## Fixed evaluation protocol

- Primary ROI: within 30 m horizontally, within the known scene ground footprint, voxel-center world height 0.3..3.0 m. This measures above-ground occupied geometry.
- Exclude an ego-frame column spanning |x| <= 2.5 m and |y| <= 1.4 m. Exclude below-ground and outside-scene voxels rather than assuming they are free.
- Score every voxel in that ROI, including occluded geometry. No camera/LiDAR visibility mask is applied.
- Raw argmax class 17 is empty; every other class is nonempty for binary occupancy. Class 0 is counted as nonempty, consistent with the upstream geometric metric, and its frequency is recorded. No confidence filtering is applied to the primary score.
- Sum TP/FP/FN/TN across frames, then compute occupied IoU, precision, recall and F1. Voxel accuracy and the all-empty accuracy baseline expose the free-space imbalance.
- Secondary results include ground (-0.25..3.0 m) and the existing fixed confidence threshold of 0.35. These do not replace the primary score, and no threshold search was performed.
- Semantic mIoU is not reported: a verified mapping from the simulator's objects to the checkpoint's semantic classes is still needed.

This filled-geometry target differs from the official dataset annotation protocol. Low scores may reflect model errors, calibration/domain mismatch, occlusion, or annotation differences; this initial experiment does not separate those causes. The definition of occupied IoU follows scene-completion evaluation, but these scores must not be compared directly with published [SurroundOcc benchmark results](https://openaccess.thecvf.com/content/ICCV2023/papers/Wei_SurroundOcc_Multi-camera_3D_Occupancy_Prediction_for_Autonomous_Driving_ICCV_2023_paper.pdf). The local checkpoint's geometric metric is implemented in `GaussianFormer3D/misc/metric_util.py`.

## Initial measured baseline — November 15, 2026

Five paired poses from one parking sequence, six 640 x 360 cameras and 11 LiDAR sweeps per pose. All **941** checkpoint tensors match strictly; weights are frozen. These five correlated frames are an initial integration baseline, not an independent generalization test set.

| Primary metric | Result |
| --- | ---: |
| Occupied IoU | 15.41% |
| Precision | 28.37% |
| Recall | 25.23% |
| F1 | 26.71% |
| Voxel accuracy | 82.28% |
| All-empty voxel-accuracy baseline | 87.21% |

An all-empty predictor has zero occupied recall and IoU here despite its higher voxel accuracy. Occupied IoU, precision and recall are therefore the meaningful primary measures.

On an RTX 5090, two warm-up forwards followed by 15 timed forwards (three per pose), batch size 1 and FP32:

| Timing scope | Median | P95 |
| --- | ---: | ---: |
| CUDA-synchronized forward + softmax/argmax | 197.3 ms | 216.8 ms |
| Input reading and preprocessing | 103.2 ms | 133.2 ms |
| Saved-input pipeline through CPU prediction output | 311.9 ms | 336.0 ms |
| Live instrumented sensor capture to accepted model result, excluding first two updates | 816.7 ms | 863.8 ms |

Offline pipeline timing excludes scene rendering, model loading, output saving and ground-truth generation. Live timing includes rendering, snapshot export, file transport and main-loop scheduling; it is recorded separately and is not a pure model benchmark. First live update took about 5.38 seconds, so warm-up matters. Peak PyTorch allocated GPU memory was **3.431 GB**, not total GPU usage. Fresh-process startup took **21.01 seconds** in this run and is excluded from warm latency. These results do not claim live 5 FPS or whole-system real-time operation.

Evidence: [full report](../reports/evaluation/20261116T001548_636462Z/REPORT.md), [accuracy](../reports/evaluation/20261116T001548_636462Z/accuracy.json), [latency](../reports/evaluation/20261116T001548_636462Z/latency.json), and [comparison figure](../reports/evaluation/20261116T001548_636462Z/occupancy-comparison.png). UTC-based directory names can fall on November 16 while the local evaluation date is November 15.

## Verification and next experiments

The deterministic evaluation assignment change is measured separately in
[anchor assignment results](../reports/anchor_assignment_fix/RESULTS.md).
Historical scores above and the earlier ablation reports used randomized
assignment and remain archived under that policy. Reproduce the new comparison
from WSL after dependency setup:

```bash
bash tools/evaluate_anchor_assignment.sh
python tools/summarize_anchor_assignment.py
```

The exact multiview ablation baseline (realistic cars, building present,
640x360, eleven sweeps) also has a separate 20-repeat comparison at each of
the four original viewpoints:
[baseline IoU rerun](../reports/anchor_assignment_baseline/RESULTS.md).
Reproduce it with `bash tools/evaluate_anchor_baseline.sh` in WSL. It compares
fresh randomized and deterministic inference against the archived baseline,
using identical saved sensor inputs and raw-argmax scoring.

The comparison replays four existing viewpoints with simple and realistic cars,
uses unchanged raw-argmax geometric accuracy scoring, and measures repeatability
with the unchanged overlay filters. It also checks ten actual consecutive
stopped captures. Restart the scene's model worker to load the updated inference
code; an already running worker keeps its previously imported implementation.

Nine numerical tests cover thin-object intersection, solid interiors, triangle intersection rejection, coordinate rotation/translation, ROI exclusion, confusion arithmetic, undefined denominators, perfect predictions and the all-empty baseline. Run them in WSL after sourcing `tools/linux_env.sh`:

```bash
python -m pytest -s -q tools/tests/test_occupancy_evaluation.py
```

Next, expand to independently varied scenes and held-out trajectories; add an explicitly defined visibility-aware or surface-occupancy protocol alongside filled geometry; verify semantic class mapping before reporting semantic mIoU; and compare controlled calibration/sensor settings. Keep the current protocol as the fixed baseline so later changes remain interpretable. Hardware/resource comparisons remain separate course deliverables.
