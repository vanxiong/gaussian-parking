# Working validation-data sources

All datasets, transfer caches, dependencies and reports remain inside `gaussian_parking`. `evaluation/download_surroundocc_data.bat` installs/resumes the same public-source acquisition pipeline. It does not require account credentials. Do not start a second copy while a download is active.

| Component | Working source | Installation |
| --- | --- | --- |
| nuScenes trainval metadata | [Official Motional AWS object](https://motional-nuscenes.s3.ap-northeast-1.amazonaws.com/public/v1.0/v1.0-trainval_meta.tgz) | `data/nuscenes/v1.0-trainval/` |
| RGB keyframes and LiDAR | [Official Motional AWS distribution](https://registry.opendata.aws/motional-nuscenes/) | Only the referenced validation files are retained under `data/nuscenes/samples/` and `sweeps/` |
| SurroundOcc validation camera/pose/calibration metadata | [Researcher-published Tsinghua download](https://cloud.tsinghua.edu.cn/d/bb96379a3e46442c8898/files/?p=%2Fnuscenes_infos_val_sweeps_occ.pkl&dl=1) | Converted to `data/nuscenes_cam/nuscenes_infos_gf3d_val.pkl` |
| Sparse projected depth | [DFA3D authors' depth archive](https://drive.google.com/file/d/1C0lYFU1Wu2fVOjC-YcUBThS5rf4ehieR/view) | Only 36,114 validation image depth records under `data/nuscenes/depth_gt/` |
| SurroundOcc occupancy | [Original-format mirror](https://huggingface.co/datasets/sangramrout/surroundOCC/tree/main) | Only 6,019 validation `.pcd.bin.npy` label files under `data/surroundocc/samples/` |

The working metadata link is documented by the [GaussRender researchers](https://github.com/valeoai/GaussRender). The working depth link is documented in the [DFA3D preparation instructions](https://github.com/IDEA-Research/3D-deformable-attention#installation). These replace unavailable GF3D SharePoint links. The original Tsinghua validation-label ZIP link returns “Link does not exist,” so the label mirror is used instead. UniOcc's reformatted occupancy dataset was inspected but is not substituted for original-format annotations.

## Validation and provenance

- The downloaded validation metadata has the required `infos`/`metadata` schema, 6,019 unique samples and the official 150-scene split. Full sample-token membership is checked against the original nuScenes tables and the pinned official scene list.
- That metadata does not contain GF3D LiDAR histories. The tracked `GaussianFormer3D/tools/make_gf3d_infos.py` helper's `obtain_sensor2top` routine generates ten previous sweeps from original nuScenes sample-data, calibration and ego-pose records. This is regenerated metadata, not a byte-identical copy of the missing GF3D release pickle. Camera records are retained; paths are relocated. Details and the helper hash are recorded in `reports/surroundocc/sweep-regeneration.json`.
- All 58,664 generated sweep transformations were checked against independent homogeneous pose composition. Maximum absolute discrepancy: approximately `1.36e-12`.
- Sensor archives are verified against the official S3 object ETags before extraction. The downloaded metadata archive differs from the legacy `md5.checksum` entry; its current S3 multipart ETag `fd4ea76d8701fb567a67a65025d0b022-56` matches exactly. This discrepancy is recorded rather than ignored.
- The four mirrored label archives are verified against the Hugging Face LFS SHA-256 values recorded in `reports/surroundocc/label-mirror-files.json`. All required validation filenames must be present. These checks verify transfer integrity and coverage; an official original-label archive checksum was not available for cross-mirror byte comparison.
- Selected depth files are checked by ZIP CRC during extraction and must be float32 x/y/depth triples. Every validation image must have its corresponding depth filename. Depth comes from real nuScenes LiDAR projection, not simulator rendering.
- Archive members are selected by the required filenames, confined beneath `gaussian_parking/data`, written via temporary files, and checked for completeness. Existing files of the wrong size are preserved and trigger an error. Four numerical tests exercise extraction safety and incomplete-file handling.

The complete sensor selection is **94,522 files**. Twenty official archives are read (ten keyframe and ten LiDAR archives), because the original provider bundles training and validation together. The downloader keeps only the selection and releases each verified sensor archive afterward. Cached depth/label and metadata downloads remain available for reproducibility. Per-archive receipts and transfer logs are under `reports/surroundocc/`. Disk checks prevent continuing below the reserved space floor.

After acquisition, run `evaluation/evaluate_surroundocc.bat`. The data preflight must pass before CUDA is loaded; a small `--limit` run is only a diagnostic, not the full standard baseline.
