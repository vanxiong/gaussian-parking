# GaussianFormer3D — parking and street scenes

For the optional all-parked-car visual ablation, use `drive_realistic_cars.bat` or
`--parked-car-appearance realistic`; `simple` remains the default. See
[car appearance ablation](docs/CAR_APPEARANCE_ABLATION.md) for stationary evaluation
commands and the shared occupancy-reference protocol.

The optional `--building absent` scene toggle removes the building. Run
`evaluation\evaluate_building_ablation.bat --repeats 20 --anchor-assignment random` for the paired comparison
with all ten realistic parked cars; see [building ablation](docs/BUILDING_ABLATION.md).

The [consolidated stationary ablation report](reports/ablation_study/20261118T055357_977940Z/REPORT.md)
contains the four completed comparisons (appearance, building, native resolution,
LiDAR history), 540 independently rescored predictions across the initial study and four-viewpoint extension, figures and source data.
For the two sensor comparisons, run `evaluation\evaluate_sensor_ablations.bat
--baseline-report <realistic-building-present-640-report> --repeats 20`.

The active project has two scene launchers:

For a fresh clone, follow [installation](docs/INSTALLATION.md). Isaac Sim must be
installed separately and linked using `maintenance/link_isaac_sim.ps1`.

| Launcher | Scene |
| --- | --- |
| **drive_gaussianformer3d.bat** | Start in the parking lot and park automatically |
| **drive_street_gaussianformer3d.bat** | Drive from the street into the parking destination |

Double-click either BAT. Both run directly through Isaac Sim, a frozen GaussianFormer3D worker in WSL Ubuntu 24.04, and the CPU planner. They mount the faster dependency snapshot automatically. No ROS launch is required. The model stays loaded during a scene session and exits when that session closes.

The GUI provides stop/resume, replan, reset, destination heading and obstacle controls. The default speed cap is 1 m/s. Yellow markers show simulated LiDAR evidence; purple markers show predicted occupancy. LiDAR supplies hard clearance checks, the model supplies semantic route costs, and simulator poses provide localization. Successful parking does not establish occupancy accuracy or real-car readiness.

Both driving scenes also open a **Sensor views | surround cameras + LiDAR** window.
Six labeled camera tiles show the latest synchronized model input: front-left,
front, front-right above back-left, back, back-right. They refresh at the existing
perception capture rate, with capture age displayed separately from live LiDAR age.
The right-hand plot shows the latest raw scan relative to the car (forward up,
30 m range). Rings mark distance; gray
returns are low, yellow returns are above 0.25 m, and blue returns are above 2 m
relative to the car's origin. Colors indicate height, not semantic classes.
Use **Show surround cameras + LiDAR** in car settings to reopen it after closing it.
Both GUI launchers start with a 1920 × 1400 application window and dock sensors
below the viewport, controls at lower right, and status between Stage and controls.
The sensor image scales to fit its panel. **Restore driving layout** reapplies
these positions after manual rearrangement.

Fourteen open bays are numbered on the pavement: spots 1–8 along the near row
and 9–14 along the building row, in increasing world X. The default destination
is spot **8**, between the red and white cars in the near row. In car settings,
choose a bay and press **Park in selected spot** to brake and replan from the
current pose. The selected bay is outlined in green. These are open bays in the
authored layout; sensor clearance checks still apply if a bay becomes blocked.
Use `--parking-spot 11` to choose the previous building-row destination on launch.
Choose **Front-in** or **Back-in** under **Parking style**, then press **Park in
selected spot** to apply both choices. Back-in turns the final heading toward the
aisle and requires a reverse final entry; approach maneuvers may use either gear.
The default remains front-in. Use `--parking-mode back-in` on either launcher to
start with reverse parking selected.

Perception targets 2 captures per simulation second by default. Only one request
is processed at a time, so a slower worker reduces the actual rate instead of
building a queue. Pass `--prediction-hz 1` to either launcher to restore the
previous rate. Higher capture rates add rendering, transfer and processing work.

## Folder layout

- `simulation/`: scene, vehicle, sensing, planning and control source; `output/` holds the required parking USD and camera calibration.
- `tools/`: internal model workers, preprocessing, evaluation and setup helpers.
- `GaussianFormer3D/`, `checkpoints/`, `vendor/`, `.toolchains/`, `.venv/`, `.venv-linux/`, `.runtime-linux/`: model/runtime dependencies, not orphan folders.
- `runtime/`: ignored, machine-local junction to a separately installed Isaac Sim.
- `evaluation/`: preserved capture, scoring and dataset-download launchers.
- `data/`, `simulation/data/`, `reports/`: preserved downloads, partial transfers, captures, predictions and measurement results. The SurroundOcc download remains paused.
- `maintenance/`: environment checks, rebuild and scene-generation launchers.
- `project_materials/`: local course materials and historical records, excluded from Git and not required to run the project.

The repository does not depend on the historical `current_version` folder.

## Preserved evaluation work

From this directory:

```powershell
# Evaluate an existing paired Isaac Sim capture run
.\evaluation\evaluate_gaussianformer3d.bat

# Inspect standard-dataset readiness without starting downloads
.\.venv\Scripts\python.exe tools/check_surroundocc_data.py

# Resume acquisition only when you want to continue
.\evaluation\download_surroundocc_data.bat

# Run standard validation after data is complete
.\evaluation\evaluate_surroundocc.bat
```

For paired ground-truth captures, add `--evaluation-snapshots` to either scene launcher. See [Isaac Sim evaluation](docs/OCCUPANCY_EVALUATION.md), [standard validation](docs/SURROUNDOCC_EVALUATION.md) and [download sources](docs/SURROUNDOCC_DOWNLOAD_SOURCES.md).

Weights remain frozen, with 941 strict checkpoint tensors. The cleanup inventory and verification results are under `reports/cleanup/`.

Repeat all ablations across four fixed viewpoints using `evaluation\evaluate_multiview_ablations.bat --repeats 20 --anchor-assignment random` (400 new predictions).
See [reproducing ablations](docs/ABLATIONS.md) for fresh captures, explicit random
or deterministic anchor policies, saved-input comparisons and ignored outputs.
