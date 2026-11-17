# GaussianFormer3D — parking and street scenes

The active project has two scene launchers:

For a fresh clone, follow [installation](docs/INSTALLATION.md). Isaac Sim must be
installed separately and linked using `maintenance/link_isaac_sim.ps1`.

| Launcher | Scene |
| --- | --- |
| **drive_gaussianformer3d.bat** | Start in the parking lot and park automatically |
| **drive_street_gaussianformer3d.bat** | Drive from the street into the parking destination |

Double-click either BAT. Both run directly through Isaac Sim, a frozen GaussianFormer3D worker in WSL Ubuntu 24.04, and the CPU planner. They mount the faster dependency snapshot automatically. No ROS launch is required. The model stays loaded during a scene session and exits when that session closes.

The GUI provides stop/resume, replan, reset, destination heading and obstacle controls. The default speed cap is 1 m/s. Yellow markers show simulated LiDAR evidence; purple markers show predicted occupancy. LiDAR supplies hard clearance checks, the model supplies semantic route costs, and simulator poses provide localization. Successful parking does not establish occupancy accuracy or real-car readiness.

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
