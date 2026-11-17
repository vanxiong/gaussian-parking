# Evaluation tools

These launchers were moved out of the main scene directory. Evaluation scripts, reports, datasets and partial downloads remain at their original paths, so saved manifests remain usable.

Run from `gaussian_parking`:

```powershell
.\evaluation\evaluate_gaussianformer3d.bat
.\evaluation\evaluate_surroundocc.bat
.\evaluation\setup_surroundocc.bat
```

`download_surroundocc_data.bat` resumes the paused dataset transfer; it does not run automatically. `capture_cameras.bat` and `capture_lidar.bat` retain the optional stationary input capture workflow.

See `../docs/OCCUPANCY_EVALUATION.md`, `../docs/SURROUNDOCC_EVALUATION.md`, and `../docs/SURROUNDOCC_DOWNLOAD_SOURCES.md` for scoring protocols and input requirements.
