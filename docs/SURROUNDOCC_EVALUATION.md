# Standard SurroundOcc validation baseline

**Working sources have been resolved and acquisition is implemented.** Use `evaluation/download_surroundocc_data.bat`; see `reports/surroundocc/STATUS.json` for the current data/evaluation state. The downloaded metadata matches all 6,019 official validation samples, labels and depth are installed, and sensor transfers are tracked in `reports/surroundocc/sensor-download-progress.json`. The full benchmark score is separate from download/preflight success.

See [working source links and verification receipts](SURROUNDOCC_DOWNLOAD_SOURCES.md). The broken GF3D SharePoint metadata URL is replaced with the researcher-published SurroundOcc metadata plus the tracked GF3D preparation routine for LiDAR histories. Sparse depth comes from the original DFA3D archive. Transfer integrity, native formats, split membership and generated sweep transforms are verified. No account credentials are required for these acquisition routes.

All source, evaluation dependencies, dataset files, caches and reports are scoped to `gaussian_parking`. Existing Isaac Sim evaluation and model weights are preserved. The standard benchmark is a separate result from our custom Isaac Sim filled-geometry evaluation.

## Obtain the validation inputs

1. Obtain sensor archives through the official [anonymous AWS distribution](https://registry.opendata.aws/motional-nuscenes/) or register/sign in at the [nuScenes download page](https://www.nuscenes.org/download). The AWS bucket listing was verified without credentials; exact archive names and sizes are saved in `reports/surroundocc/aws-archives.json`. Account registration is only needed for the website route; no credentials are stored by these tools.
2. Obtain nuScenes v1.0 trainval metadata and the camera/keyframe LiDAR and historical LiDAR sweep files referenced by the validation metadata. Full-dataset archive sizes are large: check archive/extraction space before downloading. Preflight reports actual free space and, once metadata is present, enumerates the exact required files. Training sensor data is not needed for inference, but the provider may bundle train and validation files together.
3. Obtain the GaussianFormer3D authors' **validation metadata** and **precomputed depth inputs** from their [official preparation instructions](https://github.com/lunarlab-gatech/GaussianFormer3D#nuscenes-dataset-preparation). These are separate from the nuScenes sensor archives. The generic SurroundOcc/BEVFormer pickle is not the authors' metadata schema.
4. Obtain the **200 x 200 x 16 validation occupancy labels** from the [SurroundOcc data instructions](https://github.com/weiyithu/SurroundOcc/blob/main/docs/data.md). The listed validation label archive is 627 MB; the full mesh-vertex download is not required.

The professor's [GaussianFormer3D study guide](https://github.com/lkk688/DeepDataMiningLearning/blob/main/DeepDataMiningLearning/ngperception/docs/GaussianFormer3D_study.md) confirms these input requirements. The authors' [public issue #3](https://github.com/lunarlab-gatech/GaussianFormer3D/issues/3) reports multiple expired download links. The replacement sources and documented metadata regeneration now resolve that earlier access problem.

Expected layout:

```text
gaussian_parking/
  data/
    nuscenes/
      samples/CAM_FRONT/ ... five other cameras and LIDAR_TOP/
      sweeps/LIDAR_TOP/
      depth_gt/<camera-image-filename>.jpg.bin
      v1.0-trainval/scene.json
      v1.0-trainval/sample.json
    nuscenes_cam/nuscenes_infos_gf3d_val.pkl
    surroundocc/samples/<lidar-filename>.pcd.bin.npy
```

The sensor files retain their original filenames. Depth files contain the authors' float32 x/y/depth triples; dense depth arrays or simulated depth are not equivalent inputs. Metadata contains historical sweep paths, so copying just current keyframes is insufficient.

## Commands

From `gaussian_parking`:

```powershell
.\evaluation\download_surroundocc_data.bat
.\evaluation\setup_surroundocc.bat
.\.venv\Scripts\python.exe tools/check_surroundocc_data.py
.\evaluation\evaluate_surroundocc.bat
```

Setup installs only `pyquaternion==0.9.9` into `.eval-deps` (without dependencies); it does not upgrade the model venv or modify its read-only dependency image. Paths and reports stay inside this version. Data is not committed to Git.

Preflight requires **6,019 unique validation samples**, checks sample-token and scene membership against the official **150-scene validation split** using nuScenes `scene.json`/`sample.json`, and enumerates every required camera, depth, occupancy, keyframe LiDAR and possible historical sweep file. Missing files or an incorrect split prevent CUDA/model startup. Details are written to `reports/surroundocc/data-preflight.json`. The official validation scene names and source hash are pinned in `configs/surroundocc_val_scenes.json`.

When data is complete, the full run writes `reports/surroundocc/<timestamp>/results.json`, per-frame progress, class IoUs, geometric IoU, semantic mIoU, checkpoint/data hashes and latency information. `reports/surroundocc/latest.json` identifies the latest completed result. A run with `--limit N` is explicitly labeled a partial diagnostic and never a complete standard validation result. Preflight still checks the complete dataset for that diagnostic.

## Protocol fidelity

- Uses the authors' unchanged validation dataset and transforms from `GaussianFormer3D/dataset/` and the checked-in `nuscenes_surroundocc_gs25600.py` configuration. Official camera resolution, RGB normalization, padding, projection matrices, precomputed depth and sweep transforms are retained. Paths are relocated to the project data folder.
- Preserves the authored sweep-selection settings, including seeded random selection when more than ten sweeps exist, and the source's timestamp arithmetic. Those behaviors differ from the Isaac Sim adapter; they are not silently corrected while reproducing the authors' evaluation.
- A small adapter implements the `BasePoints` tensor/index/new_point/cat operations exercised by that pipeline because the current inference runtime has no MMDetection3D installation. The actual upstream transforms are exercised in tests. This is not a general MMDetection3D replacement. The real dataset/GPU combination remains unverified until data is supplied.
- Loads the existing checkpoint with **941 strictly matched tensors**; all parameters remain frozen. No training, optimizer or gradient updates occur. Occupancy labels stay on CPU and are removed from network inputs; the head queries the same geometric grid.
- Scores the complete **200 x 200 x 16 LiDAR-frame grid**, 0.5 m voxels, bounds `[-50,-50,-5]` to `[50,50,3]`. No simulator ROI or height/visibility filter is applied.
- Semantic mIoU averages the 16 classes numbered 1..16. Empty is 17. The primary `author_compatible` result reproduces `GaussianFormer3D/misc/metric_util.py`, including its absent-GT-class convention and geometric counting of class 0 as nonempty. The author's checked-in SurroundOcc evaluator does not apply the loader's class-0 mask. An explicitly separate `diagnostic_ignore_gt_zero` result excludes GT class 0 for inspection.
- The source `eval.py` updates its metric twice per sample in this case. The runner counts each sample once; uniform doubling does not change IoU ratios.
- Timing is CUDA-synchronized forward, finite-logit checking, argmax and CPU label copying. It excludes input loading/transfers and is not the same scope as our previous simulator latency table. All-pass and warm median information are labeled; hardware and allocated CUDA memory are recorded.

## Checks completed before data access

Eight tests verify exact metric parity against the author's actual `MeanIoU` class source, sparse label conventions, invalid annotation rejection, confined path relocation, class-0 diagnostic separation, official scene-list size, sweep transforms/timestamp arithmetic, and an end-to-end pass through the actual upstream validation loader using synthetic camera/LiDAR/depth/occupancy fixtures. The loader test checks RGB normalization, padding, query coordinates and sweep padding. **These tests do not produce a measured SurroundOcc baseline.**

Run in WSL after sourcing `tools/linux_env.sh`:

```bash
python -m pytest -s -q tools/tests/test_surroundocc_validation.py
```

The author-reported checkpoint reference is **43.3% occupied IoU / 27.1% semantic mIoU**, from the [official model table](https://github.com/lunarlab-gatech/GaussianFormer3D#evaluation). These values are published references, not reproduced results from this computer. The first full run must be compared against them before interpreting any change to the input pipeline or using the official score alongside Isaac Sim results.
