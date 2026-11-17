# Installation on Windows

Isaac Sim is an external prerequisite, not bundled in this repository. You can
clone this project anywhere; no `current_version` directory is needed.

## Isaac Sim

Use an existing standalone Windows installation, or download and extract one
outside this repository using NVIDIA's [download page](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html)
and [workstation installation guide](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html).
Follow NVIDIA's license and hardware/driver requirements. The installation root
must contain `python.bat` and `kit/`.

The existing development installation reports **6.1.0-rc.26**. Other versions
have not been validated by this project; a newer stable release may require API
adjustments. Keep at least **50 GB free** when installing, plus space for model
environments, weights, evaluation data and simulator caches.

From the cloned repository, link your installation:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\maintenance\link_isaac_sim.ps1 -IsaacSimPath "C:\isaacsim"
.\runtime\python.bat -c "import sys; print(sys.version)"
```

This creates an ignored Windows junction named `runtime`. Re-run the command to
select a different installation. It refuses to overwrite a real runtime folder.
The link is machine-local and is not included in Git.

## Model dependencies and scene launch

Isaac Sim alone is not sufficient for model inference. Install WSL Ubuntu 24.04
with GPU access, Python 3.12, Git and the build prerequisites described in
`tools/bootstrap_linux.sh`. Restore the pinned model/vendor sources using
`python tools/restore_sources.py`; see [source dependencies](../dependencies/README.md).
Use `maintenance/build_environment.bat` for the Linux model environment and CUDA
extensions. Download frozen weights with `tools/download_checkpoint.py` in an
Python environment. Evaluation datasets are separate;
see the evaluation documentation. These are substantial external dependencies,
not files that should be committed.

The small parking scene and camera calibration are included. After dependencies
are ready, run `drive_gaussianformer3d.bat` or
`drive_street_gaussianformer3d.bat`. First startup may download NVIDIA assets and
compile shaders. Setup checks do not establish GPU or driving correctness; use
the documented environment, CUDA and scene validation procedures.
