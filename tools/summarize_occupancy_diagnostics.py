"""Summarize measured diagnostic outputs without changing evaluation policies."""
import argparse,json
from pathlib import Path
import numpy as np

CLASSES=['noise','barrier','bicycle','bus','car','construction_vehicle','motorcycle','pedestrian','traffic_cone','trailer','truck','driveable_surface','other_flat','sidewalk','terrain','manmade','vegetation','empty']

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);a=p.parse_args()
    root=a.manifest.parent;out=root/'diagnostics';d=json.loads((out/'diagnostics.json').read_text());probes=json.loads((out/'input-probes.json').read_text())
    audit=json.loads((root.parents[1]/'calibration-audit.json').read_text())
    repeat=json.loads((out/'repeatability.json').read_text())
    b=d['baseline'];building=d['ground_truth_group_recall']['building'];other=d['ground_truth_group_recall']['other_geometry'];free=d['lidar_free_ray_evidence']
    pct=lambda n:f'{100*n:.2f}%'
    lines=['# Stationary occupancy diagnosis','',
        'The evidence does not support a global calibration or coordinate-frame mistake. Low IoU persists with verified scene transforms; predictions occupy measured free space, while filled, partly hidden building geometry dominates missed occupancy. This conclusion applies to this single captured scene.',
        '',f"Reproduced baseline: IoU **{pct(b['occupied_iou'])}**, precision **{pct(b['precision'])}**, recall **{pct(b['recall'])}**; TP {b['tp']}, FP {b['fp']}, FN {b['fn']}, TN {b['tn']}.",
        '', '## Calibration and coordinates','',
        f"- USD ego transform error: {audit['world_from_ego_max_error']}; LiDAR extrinsic error: {audit['lidar_extrinsic_max_error']}.",
        f"- Across six cameras, maximum USD-versus-metadata extrinsic error: {max(c['extrinsic_max_error'] for c in audit['cameras'])}; intrinsic error: {max(c['intrinsic_max_pixel_error'] for c in audit['cameras']):.8f} pixels. All aperture offsets are zero.",
        '- Model preprocessing pairs each named RGB image with its own calibration, composes optical-from-ego with ego-from-LiDAR, normalizes RGB, and pads only bottom/right. These conventions match the checked upstream test pipeline. The model query grid and scorer both use XYZ, 0.5 m cells, range [-50,-50,-5] to [50,50,3].',
        f"- All {d['lidar_geometry_surface_check']['current_returns']} current LiDAR returns have convex-surface plane residual <=5 cm against exported geometry. Median residual is {d['lidar_geometry_surface_check']['median_plane_residual_m']:.3g} m. This residual is not an exact Euclidean corner distance.",
        '- Projected LiDAR returns were visually inspected over all six camera images. They align with scene structures; some behind-object returns are expected from the LiDAR/camera baseline and collider/render differences. This is qualitative supporting evidence, not a pixelwise calibration accuracy metric.',
        '', '![LiDAR camera projections](lidar-camera-projections.png)',
        '', '## Alignment stress tests','', '| Transformation | IoU |','| --- | ---: |']
    for row in d['alignment_tests']:
        label='Z reversal' if row.get('flip_z') else f"swap XY={row['swap_xy']}, flip X={row['flip_x']}, flip Y={row['flip_y']}"
        lines.append(f"| {label} | {pct(row['occupied_iou'])} |")
    yaw=max(d['yaw_tests'],key=lambda r:r['occupied_iou']);shift=d['best_local_translations'][0]
    lines+=['',f"Best tested yaw among -10,-5,-2,0,2,5,10 degrees: {yaw['yaw_deg']} degrees, IoU {pct(yaw['occupied_iou'])}.",
        f"Best tested translation within +/-2 m in 0.5 m steps: {shift['offset_m']} m, IoU {pct(shift['occupied_iou'])}. This is an optimistic ground-truth-selected diagnostic, not a correction. It shifts {d['translation_roi_effect']['original_false_positives_shifted_out_of_roi']} original false positives outside the evaluation ROI. Thus its apparent gain cannot be interpreted as proof of misregistration.",
        '', '## What contributes to the errors?','',
        f"- {free['false_positives_in_these_cells']} / {b['fp']} false positives ({pct(free['false_positives_in_these_cells']/b['fp'])}) lie in geometry-empty cells traversed by measured LiDAR rays, after a 0.5 m pre-hit margin. This supports genuine excess predicted occupancy. Rays only sample parts of cells; this does not replace the primary protocol.",
        f"- Building geometry contributes {building['occupied_voxels']} / {b['ground_truth_occupied_voxels']} occupied cells ({pct(building['occupied_voxels']/b['ground_truth_occupied_voxels'])}) and {building['missed']} / {b['fn']} false negatives ({pct(building['missed']/b['fn'])}). Building recall: {pct(building['recall'])}; remaining geometry recall: {pct(other['recall'])}. These are geometry recall groups, not semantic evaluation.",
        '- The primary target fills object interiors and includes occluded voxels. That makes hidden building completion a large part of this score; it is different from measuring visible surface reconstruction.',
        '', '| Predicted class of false-positive cells | Cells | Share of FP |','| --- | ---: | ---: |']
    for c,n in sorted(d['false_positive_classes'].items(),key=lambda r:r[1],reverse=True):lines.append(f'| {CLASSES[int(c)]} | {n} | {pct(n/b["fp"])} |')
    lines+=['','Labels describe the model output only; simulator semantic labels have not been verified. Road/sidewalk predictions at above-ground heights remain errors under the fixed geometric protocol. Removing these labels would change the task, not fix calibration.',
        '', '| Range | IoU | Recall |','| --- | ---: | ---: |']
    for r in d['range_bands']:lines.append(f"| {r['range_m'][0]}–{min(30,r['range_m'][1])} m | {pct(r['occupied_iou'])} | {pct(r['recall'])} |")
    lines+=['', '## Controlled model input probes','',
        'Each variant uses seeds 42,43,44, with unchanged ground truth and scoring. Intensities 0/64/128 replace the native constant 255 and are hypothetical sensitivity probes, not calibrated reflectance. Image upsampling scales intrinsics consistently but adds no scene detail.',
        '', '| Variant | Mean IoU | Min–max IoU |','| --- | ---: | ---: |']
    for name in dict.fromkeys(r['variant'] for r in probes['runs']):
        values=[r['occupied_iou'] for r in probes['runs'] if r['variant']==name]
        lines.append(f'| {name} | {pct(np.mean(values))} | {pct(min(values))}–{pct(max(values))} |')
    rows=repeat['runs'];values=[r['occupied_iou'] for r in rows]
    lines+=['',f"Three identical-input runs, each resetting the seed to 42: IoU {pct(min(values))}–{pct(max(values))}. Minimum all-grid class-label agreement to the first run: {pct(min(r['all_voxel_label_agreement_to_first'] for r in rows))}. Minimum ROI binary-occupancy agreement: {pct(min(r['roi_occupancy_agreement_to_first'] for r in rows))}.",
        '', 'The LiDAR lifter uses torch.randperm during evaluation. Residual same-seed differences also demonstrate runtime nondeterminism; the responsible kernel was not isolated. Warm-up forwards advance random state, so live capture and offline replay need not produce identical maps. These effects do not account for a low score persisting across all probes.',
        '', '## Recommended next work','',
        '1. Keep the current transform chain and primary baseline; do not apply ground-truth-selected shifts or intensity settings.',
        '2. Make evaluation repeatability explicit: fixed anchor assignment or saved RNG state per sample, then isolate remaining nondeterministic operations. Report repeated-run spread until that is resolved.',
        '3. Add a separately named visible-surface or visibility-aware score alongside full solid completion, so hidden building interiors do not obscure performance on observable obstacles.',
        '4. Test additional stationary positions and native higher-resolution captures. Upsampling this saved frame did not help. Investigate calibrated LiDAR reflectance and synthetic-to-real appearance mismatch with held-out scenes; these remain hypotheses, not established causes.',
        '', '## Inspectable evidence','',
        '- `diagnostics.json`: exact confusion counts, transforms stress tests, range/height/class cuts, free-ray check.',
        '- `input-probes.json`, `repeatability.json`: every model probe and repeat.',
        '- `../../../../tools/diagnose_occupancy.py`, `probe_occupancy_inputs.py`, `audit_capture_calibration.py`: computations; source capture and prediction paths remain in the parent manifest.',
        '- `../../../calibration-audit.json`: independent captured-USD parameter audit.',
        '', 'No weights, input defaults, geometric scoring policy, or original evaluation outputs were changed.']
    (out/'DIAGNOSIS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(out/'DIAGNOSIS.md')

if __name__=='__main__':main()
