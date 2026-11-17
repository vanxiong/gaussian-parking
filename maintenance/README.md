# Setup and rebuilding

The two main scene launchers use the existing installed environments and scene assets. These maintenance commands are only needed for troubleshooting or rebuilding:

- `check_environment.bat`: inspect the Linux GPU/model environment.
- `build_environment.bat`: reproduce the project-local model environment and CUDA extensions.
- `build_scene.bat`: rebuild the shared parking USD and camera calibration.
- `setup_fast_runtime.bat`: rebuild the read-only dependency snapshot after changing dependencies; stop users of the mount and unmount it first.

Use `link_isaac_sim.ps1 -IsaacSimPath "C:\isaacsim"` to connect a separate Isaac Sim
installation. See [installation](../docs/INSTALLATION.md); no historical baseline
folder is required.
