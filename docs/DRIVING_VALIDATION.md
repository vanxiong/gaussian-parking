# Driving integration checks — October 31, 2026

## Separate Isaac Sim installation — November 17, 2026

Both scene launchers passed headless end-to-end checks with the runtime junction
pointing outside the repository to Isaac Sim 6.1.0-rc.26. No historical baseline
installation was used. Tests reused the existing model environments and weights;
they do not validate a clean-machine installation or interactive GUI controls.

| Scenario | Result | Position error | Heading error | Final speed | Model updates |
| --- | --- | --- | --- | --- | --- |
| Parking | Parked | 0.1993 m | 0.8577 degrees | 0.00924 m/s | 21 |
| Street to parking | Parked | 0.1997 m | 0.1020 degrees | 0.00855 m/s | 74 |

Neither run recorded a collision-watchdog overlap. Every perception update used
the frozen model with 941 strict checkpoint tensors and no ground-truth labels.
Local evidence: `simulation/data/driving_runs/20261117T083129_486767Z/result.json`,
`simulation/data/driving_runs/20261117T083355_209848Z/result.json`, and
`reports/relocated-runtime/validation.json`. Large captures and logs are ignored.

The checks exposed two folder-relocation issues, now corrected: snapshot lock
verification used the old absolute project path, and local editable vendor
imports could retain the old checkout path when the optional mount was absent.
Lock verification and vendor imports now resolve against the active checkout.

## Direct workflow restoration — November 15, 2026

`drive_gaussianformer3d.bat` and `drive_street_gaussianformer3d.bat` use the direct model/planning workers. The retired ROS shortcuts were removed during cleanup; a compressed ROS source backup remains in `project_materials/history/ros2_source.zip`. The faster dependency image is mounted by `tools/prepare_runtime.bat`.

A real headless parking check (`--headless --screenshot --max-sim-seconds 120`) completed with status `parked`, position error **0.19965 m**, heading error **1.536 degrees**, final speed **0.00904 m/s**, no collision and **21** frozen GaussianFormer3D updates matching all **941** checkpoint tensors. Evidence: `simulation/data/driving_runs/20261115T230536_679760Z/result.json` and `reports/direct-workflow-restoration.json`. The direct worker exits with Isaac; cross-run ROS persistence is disabled in this workflow.

Run `drive_gaussianformer3d.bat` for parking and `drive_street_gaussianformer3d.bat` for the longer route. These launch the native PhysX vehicle controller, a CPU hybrid-A* planner, and a resident frozen GaussianFormer3D worker. All changes are in gaussian_parking.

The planner uses current simulated LiDAR for hard obstacle/free-space evidence, GaussianFormer3D for semantic soft costs, and simulator localization. Street mode adds known road connectivity. This demonstrates the combined system; it does not isolate the model's contribution. A LiDAR-only ablation and occupancy accuracy evaluation remain necessary for the project.

## Completed checks

| Scenario | Saved run | Result |
| --- | --- | --- |
| Parking | `20261031T194417_757774Z` | Parked within 0.200 m and 0.601 degrees; no collision-watchdog overlap. |
| Occupied destination | `20261031T194926_867586Z` | Rejected the blocked destination and remained stopped; no overlap. |
| Six-second model delay, four-second wall freshness limit | `20261031T195048_730424Z` | 353 stale-braking physics steps, then resumed and parked within 0.199 m / 1.986 degrees, final speed 0.009 m/s; no overlap. |
| Street to parking | `20261031T195619_699005Z` | Completed the route and parked in 74.4 simulated seconds, within 0.200 m / 0.486 degrees; final speed 0.009 m/s, no overlap. |

The first parking run used the original 0.15 m/s test-completion threshold. Subsequent tests require final parking speed below 0.03 m/s. The delayed run also recorded 538 moving physics steps while inference was pending, exercising asynchronous operation. Rendering a synchronized camera snapshot can still introduce a short wall-time hitch.

Each run's directory under `simulation/data/driving_runs/` contains the full result/trajectory, model and planner updates, and captured sensor inputs. `tools/summarize_driving.py` regenerates `reports/driving_validation.json`. The simulator ground-truth collision watchdog is retained as an experiment stop/failure condition, not a planning obstacle source. These few controlled scenarios establish integration, not general safety or real-world readiness.

## Controls and limits

The GUI supports stop/resume, replan, reset, destination heading, and adding/removing a destination blocker. Default cruising cap is 1 m/s. Keep the default for initial inspection. Model startup can take over a minute. Purple driving markers show GaussianFormer3D obstacle predictions; yellow markers show LiDAR-derived obstacle evidence. Optional stationary captures remain available under `evaluation/`; the unused standalone viewer was removed during cleanup.

ROS 2 topics, independent occupancy ground truth, accuracy scores, systematic runtime/resource benchmarks, and hardware comparisons remain project deliverables. A real vehicle would also need calibrated physical sensors and an actual localization system in place of simulator poses.
