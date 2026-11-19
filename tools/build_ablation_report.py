"""Consolidate the four completed stationary ablations into one evidence-backed report."""
import argparse,csv,hashlib,json,shutil
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
KEYS=('occupied_iou','precision','recall')
NAMES={'occupied_iou':'Occupied IoU','precision':'Precision','recall':'Recall'}

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def cell(s):return f"{100*s['mean']:.2f} ± {100*s['sample_sd']:.2f}"
def interval(d):return f"[{100*d['mean_95pct_t_interval'][0]:+.2f}, {100*d['mean_95pct_t_interval'][1]:+.2f}]"

def main():
    p=argparse.ArgumentParser();p.add_argument('--appearance',type=Path,required=True);p.add_argument('--building',type=Path,required=True)
    p.add_argument('--sensor',type=Path,required=True);p.add_argument('--input-probes',type=Path,help='Optional exploratory input-probes.json');p.add_argument('--output',type=Path);a=p.parse_args()
    paths={k:getattr(a,k).resolve() for k in ['appearance','building','sensor']}
    data={k:read(path/'comparison.json') for k,path in paths.items()}
    for name,path in paths.items():
        if not read(path/'validation.json')['passed']:raise ValueError('Unvalidated comparison: '+name)
        if data[name]['request']['repeats_per_variant']!=20:raise ValueError('Expected twenty repeats per condition')
    if len({d['checkpoint_sha256'] for d in data.values()})!=1:raise ValueError('Checkpoint differs across comparisons')
    policies={d['request'].get('anchor_assignment') for d in data.values()}
    if None in policies or len(policies)!=1:raise ValueError('Comparisons must record the same explicit anchor_assignment policy')
    anchor_assignment=policies.pop()
    output=(a.output or ROOT/'reports/ablation_study'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')).resolve()
    output.mkdir(parents=True,exist_ok=False);(output/'figures').mkdir()
    appearance,building,sensor=(data[k] for k in ['appearance','building','sensor'])
    diagnosis_path=paths['building']/'extra_occupancy';diagnosis=read(diagnosis_path/'diagnostics.json')
    if not read(diagnosis_path/'validation.json')['passed']:raise ValueError('Extra-occupancy diagnosis is not validated')
    figures={
        'appearance':paths['appearance']/'repeated-metrics.png',
        'building':paths['building']/'repeated-metrics.png',
        'sensor':paths['sensor']/'repeated-metrics.png',
        'occupancy':paths['building']/'paired-occupancy-comparison.png',
        'classes':diagnosis_path/'classes-and-lidar.png',
        'lidar':diagnosis_path/'spatial-lidar-comparison.png'}
    for name,path in figures.items():shutil.copy2(path,output/'figures'/(name+'.png'))
    effects=[
        {'name':'Car appearance: simple → realistic','batch':'appearance','first':'simple','second':'realistic','scope':'primary',
         'summary':appearance['summary'],'difference':appearance['paired_differences']},
        {'name':'Building: present → absent (whole scene)','batch':'building','first':'present','second':'absent','scope':'different_targets',
         'summary':building['summary'],'difference':building['paired_differences']},
        {'name':'Building: present → absent (common region)','batch':'building','first':'present','second':'absent','scope':'common_region',
         'summary':building['common_region_summary'],'difference':building['common_region_paired_differences']},
        {'name':'Native RGB: 640×360 → 960×540','batch':'sensor','first':'baseline','second':'high_resolution','scope':'primary',
         'summary':sensor['summary'],'difference':sensor['comparisons']['high_resolution']},
        {'name':'LiDAR history: 11 sweeps → 1 sweep','batch':'sensor','first':'baseline','second':'current_sweep','scope':'primary',
         'summary':sensor['summary'],'difference':sensor['comparisons']['current_sweep']}]
    effect_rows=[]
    for e in effects:
        for key,d in e['difference'].items():
            effect_rows.append({'comparison':e['name'],'metric':key,'scope':e['scope'],
                'baseline_mean_percent':100*e['summary'][e['first']][key]['mean'],
                'variant_mean_percent':100*e['summary'][e['second']][key]['mean'],
                'difference_pp':100*d['mean'],'interval_low_pp':100*d['mean_95pct_t_interval'][0],
                'interval_high_pp':100*d['mean_95pct_t_interval'][1],'pairs':20})
    with (output/'ablation_effects.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(effect_rows[0]));writer.writeheader();writer.writerows(effect_rows)
    condition_rows=[]
    for batch,d in data.items():
        for variant,summary in d['summary'].items():
            for key,s in summary.items():condition_rows.append({'batch':batch,'condition':variant,'metric':key,
                'mean_percent':100*s['mean'],'sample_sd_pp':100*s['sample_sd'],'min_percent':100*s['min'],'max_percent':100*s['max'],'replays':s['n']})
    with (output/'condition_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(condition_rows[0]));writer.writeheader();writer.writerows(condition_rows)
    # Effect intervals describe variation conditional on this one pose, not independent scenes.
    fig,axes=plt.subplots(1,3,figsize=(14,5.3),layout='constrained')
    labels=['Simple → realistic cars','Remove building: whole scene*','Remove building: common region','640×360 → 960×540 native','11 sweeps → current sweep']
    for ax,key in zip(axes,KEYS):
        for y,e in enumerate(effects):
            d=e['difference'][key];mean=100*d['mean'];lo,hi=np.array(d['mean_95pct_t_interval'])*100
            ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt='o',color='#b85d3e' if e['scope']=='different_targets' else '#326b9c',capsize=4)
        ax.axvline(0,color='#525a63',lw=1);ax.set(yticks=np.arange(len(labels)),yticklabels=labels if key==KEYS[0] else [],xlabel='Variant − baseline (percentage points)',title=NAMES[key]);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.suptitle('Stationary occupancy ablation effects | 20 paired replays per comparison\nWhiskers: conditional 95% paired t intervals; *whole-scene building targets differ')
    fig.savefig(output/'figures/effect-summary.png',dpi=180);plt.close(fig)
    date=datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-8))).strftime('%B %d, %Y')
    total=sum(len(d['runs']) for d in data.values())
    controls=building['controls'];s=sensor['summary'];timing=sensor['timing_summary']
    speed_gain=100*(1-timing['current_sweep']['median_inference_ms']/timing['baseline']['median_inference_ms'])
    app_iou=appearance['paired_differences']['occupied_iou'];res_iou=sensor['comparisons']['high_resolution']['occupied_iou'];history_iou=sensor['comparisons']['current_sweep']['occupied_iou']
    lines=['# GaussianFormer3D stationary occupancy ablation study','',
        f'Prepared {date} · Isaac Sim parking-lot experiment · Frozen nuScenes–SurroundOcc checkpoint · NVIDIA RTX 5090',
        '', '## Executive Summary','',
        f'This study completed the four planned environmental/input ablations: parked-car visual realism, building presence, native camera resolution, and LiDAR history length. It uses **{total} independently rescored saved predictions**, comprising 20 replays of each of seven condition/batch combinations at one fixed vehicle pose. It evaluates occupancy reconstruction, not driving or parking success. [1–3]',
        '',f'**Realistic parked-car visuals gave a modest occupancy benefit at this pose:** IoU increased from {100*appearance["summary"]["simple"]["occupied_iou"]["mean"]:.2f}% to {100*appearance["summary"]["realistic"]["occupied_iou"]["mean"]:.2f}% ({100*app_iou["mean"]:+.2f} percentage points). Precision increased while recall decreased. **Native 960×540 inputs were worse than 640×360** for this frozen model and scene: IoU changed by {100*res_iou["mean"]:+.2f} points, with substantially lower precision. Higher image resolution therefore did not repair the observed occupancy errors. [1,3]',
        '',f'**Removing the building did not improve whole-scene IoU.** It changed the target from {controls["ground_truth_occupied_voxels"]["present"]:,} to {controls["ground_truth_occupied_voxels"]["absent"]:,} occupied cells. On identical targets outside the removed volume, IoU improved by {100*building["common_region_paired_differences"]["occupied_iou"]["mean"]:.2f} points, but recall still fell. The large whole-scene recall increase is not evidence of better detection of the remaining objects. [2]',
        '',f'**Current-sweep-only input is an efficiency candidate, not an established accuracy improvement.** Its mean IoU change was {100*history_iou["mean"]:+.2f} points, with a conditional interval spanning zero; median model inference was {speed_gain:.1f}% faster. Recall declined slightly. The stationary scene provides little temporal viewpoint diversity, so this result should not determine the moving-vehicle configuration. [3]',
        '', 'These are controlled, single-pose findings. Seed resets do not eliminate runtime nondeterminism, the realistic cars use shared simplified occupancy/collision proxies, and no result is a reproduction of the paper’s benchmark. The recommended reference remains ten realistic visual cars, building present, 640×360 RGB and 11 sweeps; retain the other variants for controlled comparisons, not as a claim of an optimal configuration.',
        '', '## Study design and metric definition','',
        'The question is how four changes affect a frozen occupancy model in a synthetic parking lot. The ego vehicle is stationary at world XY **(−5, −2) m**, yaw **0°**, after settling. Physics is paused during synchronized capture. The driving car, checkpoint, camera placement and nonablated scene elements remain fixed. No training or tuning was performed.',
        '', '| Ablation | Baseline | Changed condition | Held fixed |','| --- | --- | --- | --- |',
        '| Car appearance | Ten simple parked cars | Ten detailed visual sedans | Positions, paint, bounds, collision/LiDAR and occupancy references; building; 640×360; 11 sweeps |',
        '| Building | Present | Entire building subtree absent | Ten realistic car visuals, other geometry, pose/calibration; 640×360; 11 sweeps. Building sensor evidence and target deliberately change. |',
        '| Camera resolution | Native 640×360 | Native 960×540 | Ten realistic cars, building present, exact LiDAR array and reference geometry; intrinsics scaled by 1.5 |',
        '| LiDAR history | Current plus ten historical sweeps | Current sweep only | Identical 640×360 RGB, calibration, current-sweep depth guidance, scene and target |',
        '', 'The primary score treats raw class 17 as empty and all other classes as occupied. Metrics are **IoU = TP/(TP+FP+FN)**, **precision = TP/(TP+FP)** and **recall = TP/(TP+FN)**. They assess geometry, not semantic-class correctness. Overall voxel accuracy is not the headline because empty cells dominate.',
        '', 'Predictions and references use the same 200×200×16 XYZ LiDAR-frame grid, 0.5 m cells, with bounds [−50,−50,−5] to [50,50,3] m. The primary evaluation covers 55,640 cells: within 30 m horizontally and the known scene footprint, world voxel-center height 0.3–3.0 m, excluding the ego XY column |x|≤2.5 m, |y|≤1.4 m. Occluded geometry is included. Ground truth is a conservative filled-solid voxelization; it is not the official SurroundOcc annotation protocol.',
        '', 'All comparisons use 20 replays with matching seeds 42–61 and two excluded warm-up forwards per condition. Two-condition studies alternate execution order; the three sensor conditions cycle all six orders. Values below are arithmetic means of per-replay ratios ± sample standard deviation. Paired differences are calculated from matched-seed runs, and 95% t intervals quantify inference variation conditional on this pose. They are exploratory, unadjusted for multiple comparisons, and not uncertainty across independent scenes. The baseline predictions in the two sensor contrasts are shared. Baselines from different batches are retained separately because nondeterminism changes their estimates.',
        '', '## Comparison of ablation effects','',
        '| Change | Δ IoU (pp) | Δ precision (pp) | Δ recall (pp) | 95% paired interval for Δ IoU (pp) |','| --- | ---: | ---: | ---: | ---: |']
    for e in effects:
        lines.append('| '+e['name']+' | '+' | '.join(f"{100*e['difference'][k]['mean']:+.2f}" for k in KEYS)+' | '+interval(e['difference']['occupied_iou'])+' |')
    lines+=['', 'The building whole-scene row uses different occupied targets; it must not be ranked as a direct same-object performance comparison. The common-region row is a secondary diagnostic from the same predictions, not a fifth independently run ablation.',
        '', '![Ablation effects with conditional intervals](figures/effect-summary.png)',
        '', '## 1. Realistic car appearance modestly improves IoU, with a recall tradeoff','',
        'All ten parked cars changed visual mesh; the ego vehicle did not change. The original car solids remained as explicit shared occupancy references and simulated LiDAR/collision shapes. Detailed surfaces are therefore not scored as exact realistic-mesh ground truth. This isolates visual appearance rather than confounding it with a new annotation or sensor shape. [1]',
        '', '| Condition | IoU (%) | Precision (%) | Recall (%) |','| --- | ---: | ---: | ---: |']
    for v,label in [('simple','Simple parked cars'),('realistic','Ten realistic visual cars')]:lines.append('| '+label+' | '+' | '.join(cell(appearance['summary'][v][k]) for k in KEYS)+' |')
    lines+=['', 'Pose, calibration, all 184,228 Nx5 LiDAR samples, projected depth inputs, reference objects and voxel targets matched exactly between the two captures. All six RGB images changed. The result supports an appearance effect at this pose, but the overall precision remains low; asset realism did not solve excess occupancy.',
        '', '![Car appearance distributions](figures/appearance.png)',
        '', '## 2. Building removal changes the task, not just its difficulty','',
        'The main structure, eight windows and canopy were removed from rendering, collision/LiDAR and ground truth together. Every nonbuilding reference object, realistic-car configuration and sensor pose remained identical. [2]',
        '', '| Whole-scene condition | IoU (%) | Precision (%) | Recall (%) |','| --- | ---: | ---: | ---: |']
    for v,label in [('present','Building present'),('absent','Building absent')]:lines.append('| '+label+' | '+' | '.join(cell(building['summary'][v][k]) for k in KEYS)+' |')
    lines+=['',f'The building accounts for {controls["removed_primary_occupied_voxels"]:,} removed occupied cells. The secondary common-region mask excludes exactly those cells from both variants and retains {controls["common_region_voxels"]:,} cells with identical truth. The mask is derived from geometry before inference, not selected using performance.',
        '', '| Common-region condition | IoU (%) | Precision (%) | Recall (%) |','| --- | ---: | ---: | ---: |']
    for v,label in [('present','Building present'),('absent','Building absent')]:lines.append('| '+label+' | '+' | '.join(cell(building['common_region_summary'][v][k]) for k in KEYS)+' |')
    lines+=['', 'Outside the removed volume, the building-free input reduces false positives enough to improve IoU, but captures slightly less of the remaining occupied geometry. In the whole-scene score, predictions in the now-empty former building volume become false positives. They average **1,473.15 cells** per replay. This explains why whole-scene and common-region IoU move in opposite directions. Removing a difficult building is not a general remedy for occupancy accuracy.',
        '', '![Building-present and absent reference/prediction slices](figures/occupancy.png)',
        '', '## 3. Higher native resolution increases recall but lowers IoU and precision','',
        'The 960×540 input was rendered natively, not enlarged from the lower-resolution image. Camera intrinsics scale correctly, extrinsics remain identical, and the LiDAR and ground-truth arrays match exactly. [3]',
        '', '| Sensor condition | IoU (%) | Precision (%) | Recall (%) |','| --- | ---: | ---: | ---: |']
    for v,label in [('baseline','640×360 RGB, 11 sweeps'),('high_resolution','960×540 RGB, 11 sweeps')]:lines.append('| '+label+' | '+' | '.join(cell(s[v][k]) for k in KEYS)+' |')
    lines+=['', 'The extra pixels led to more predicted occupancy and a precision/recall tradeoff, not better geometric agreement. This is a measured sensitivity of the frozen model and adapter in this scene. The mechanism was not isolated, and it does not establish that lower resolution is generally preferable.',
        '', '## 4. Current-only LiDAR nearly preserves IoU while reducing inference time','',
        'The current-only condition selects the 16,748 zero-lag points from the exact baseline’s 184,228 points. RGB, camera calibration, padded image dimensions and projected depth guidance stay bit-identical. The source capture still contains eleven sweeps; only the model point input is filtered. [3]',
        '', '| Sensor condition | IoU (%) | Precision (%) | Recall (%) |','| --- | ---: | ---: | ---: |']
    for v,label in [('baseline','640×360 RGB, 11 sweeps'),('current_sweep','640×360 RGB, current sweep only')]:lines.append('| '+label+' | '+' | '.join(cell(s[v][k]) for k in KEYS)+' |')
    lines+=['', 'The IoU interval spans zero, so these runs do not resolve a consistent IoU change. Precision is also unresolved; recall has a small negative paired difference. Because the car and scene are stationary, repeated sweeps contribute little viewpoint diversity. This is not evidence that temporal aggregation can be removed safely during movement.',
        '', '![Native resolution and sweep-history distributions](figures/sensor.png)',
        '', '| Condition | Median model inference | Peak allocated CUDA memory |','| --- | ---: | ---: |']
    for v,label in [('baseline','640×360, 11 sweeps'),('high_resolution','960×540, 11 sweeps'),('current_sweep','640×360, 1 sweep')]:
        t=timing[v];lines.append(f"| {label} | {t['median_inference_ms']:.1f} ms | {t['peak_cuda_allocated_GB']:.3f} GB |")
    lines+=['',f'The single-sweep median is {speed_gain:.1f}% lower than the sensor-batch baseline. Timing covers CUDA-synchronized forward plus softmax/argmax, excluding rendering, preprocessing, input transfer and final CPU copying. Memory is peak PyTorch allocation, not total GPU use. These are not end-to-end live FPS claims.',
        '', '## Excess occupancy is not primarily supported by LiDAR returns','',
        'The building-free diagnosis reuses its twenty scored predictions; it is not an additional set of independent trials. Mean false positives were **5,557.55 ± 177.16 cells**. The predicted classes were: [4]',
        '', '| Predicted label | All false positives (%) | Former building volume (%) |','| --- | ---: | ---: |']
    for name in ['driveable_surface','car','manmade','sidewalk','traffic_cone']:
        values=[]
        for scope in ['all_false_positives','former_building']:
            c=next((x for x in diagnosis['summaries'][scope]['classes'] if x['class']==name),None);values.append(100*c['share_of_false_positives'] if c else 0.)
        lines.append(f'| {name} | {values[0]:.2f} | {values[1]:.2f} |')
    lines+=['', 'No false-positive voxel contains an endpoint from any of the eleven saved sweeps. **40.35%** of all false-positive cells are crossed by current measured-return ray segments before the hits; **59.65%** lack such crossings or direct endpoints and remain unresolved by the saved rays. In the former building volume, free-ray crossings cover **17.28%**, and no false-positive center is within 2 m of an above-ground current return. No-return beams were not saved and are not assumed free.',
        '', 'Road and sidewalk labels dominate the residual former-building occupancy; it is not uniquely a hallucinated building instance. About **83.07%** of road-labeled false positives are at world height 0.3–1 m, consistent with raised/thick surface predictions. This interpretation does not prove the internal mechanism. All current LiDAR returns satisfy the 5 cm convex-surface-plane alignment diagnostic against the exported geometry.',
        '', 'A ray samples only part of a voxel. Ray crossings are supporting evidence against broad solid occupancy, not a complete visibility mask or proof that a whole voxel is empty. False-positive status is established independently from simulator geometry. Semantic names describe predictions; simulator-to-checkpoint semantic accuracy was not measured.',
        '', '![False-positive classes and LiDAR evidence](figures/classes.png)',
        '', '## What the study can support','',
        'The results support keeping a stable, documented reference configuration and treating appearance, building presence, resolution and sweep history as separate experimental factors. Use the realistic visual cars for a richer scene; keep the original shared reference geometry explicit. Keep the building as a separate challenge and report common-region diagnostics alongside whole-scene scores. Use 640×360 and eleven sweeps as the unchanged reference; treat one sweep as a speed/recall tradeoff to verify on moving or independently positioned scenes before changing a system default.',
        '', 'The next useful validation is a predefined set of independent poses/layouts, scoring the same regions for both members of each comparison. A native-resolution test at multiple poses would distinguish a scene-specific response from a broader preprocessing/model-scale sensitivity. A moving-scene sweep comparison would test whether the current-only efficiency result survives genuine temporal viewpoint and motion changes. These follow-ups were not run here.',
        '', '## Scope and limitations','',
        '- **One pose and simplified synthetic scene:** twenty replay seeds are not twenty independent scenes. No real-driving readiness, controller performance or parking success is inferred.',
        '- **Shared car proxies:** detailed car meshes affect RGB but their original simple shapes still supply LiDAR/collision and occupancy references. Scores do not establish exact detailed-mesh reconstruction.',
        '- **Changed building targets:** whole-scene comparisons have different occupied populations. Common-region results answer a different, explicitly defined question and do not replace the primary metric.',
        '- **Residual nondeterminism:** the selected anchor policy is recorded below; same-seed output variation can remain. Matching seeds and alternating order reduce confounding but do not eliminate it.',
        '- **Exploratory statistics:** conditional t intervals assume sufficiently independent paired run differences and have no multiple-comparison correction. They quantify execution variation, not generalization.',
        '- **No factorial interactions:** all four planned factors were tested, but combinations such as high resolution without the building were not. Baselines differ by batch and must not be pooled as one condition without qualification.',
        '- **No semantic mIoU or validated visibility protocol:** semantic labels are diagnostic only. Current-ray evidence is not a replacement for full-scene ground truth.',
        '- **Benchmark mismatch:** the authors report 43.3% occupied IoU and 27.1% semantic mIoU on nuScenes–SurroundOcc [6]. Their real-data benchmark uses different annotation, input and scoring conditions. That score was not reproduced here and must not be presented as directly comparable to the custom obstacle ROI.',
        '', '## Appendix A. Exploratory probes and calibration checks','',
        'Earlier checks found exact agreement of saved camera/LiDAR extrinsics with captured USD parameters and intrinsic differences below 0.000006 pixels. Axis swaps and reflections worsened the initial score; small yaw adjustments did not resolve the discrepancy. Ground-truth-selected translations partly improved scores by moving false positives outside the ROI and were not adopted. [5]',
        '', 'Earlier three-seed input probes on the original simple-car capture are retained as exploratory evidence only. They changed hypothetical constant LiDAR intensities or enlarged saved images, not calibrated reflectance or native capture. They are not mixed into the controlled twenty-replay ablations above. [5]',
        '', '| Exploratory probe on simple-car capture | Mean IoU (%) | Min–max (%) |','| --- | ---: | ---: |']
    probe_path=a.input_probes
    if probe_path is not None and probe_path.is_file():
        probe=read(probe_path)
        for name in dict.fromkeys(r['variant'] for r in probe['runs']):
            values=np.array([r['occupied_iou']*100 for r in probe['runs'] if r['variant']==name]);lines.append(f'| {name} | {values.mean():.2f} | {values.min():.2f}–{values.max():.2f} |')
    lines+=['', 'The native 960×540 experiment in the main report is distinct from this exploratory 1600×900 upsampling probe. No fabricated-intensity setting, post-hoc shift, class filter or tuned confidence threshold was selected as a fix.',
        '', '## Appendix B. Evidence, validation and reproduction','',
        f'All **{total}** formal-study prediction files were independently checked against their hashes and recomputed confusion counts, per-run metrics, means, sample SDs, ranges, paired mean differences and CSV rows. Building whole-scene and common-region scores were both checked. The LiDAR diagnosis rechecked 1,020 class/scope records; 15 analytical ray and occupancy tests passed. [1–4]',
        '', '| Source | Artifact and population |','| --- | --- |',
        f'| [1] Car appearance | [{paths["appearance"].name}](../../appearance_ablation/{paths["appearance"].name}/COMPARISON.md), 40 predictions, identical targets |',
        f'| [2] Building presence | [{paths["building"].name}](../../building_ablation/{paths["building"].name}/COMPARISON.md), 40 predictions, per-scene and common-region scopes |',
        f'| [3] Resolution/history | [{paths["sensor"].name}](../../sensor_ablation/{paths["sensor"].name}/COMPARISON.md), 60 predictions, identical targets |',
        f'| [4] Building-free diagnosis | [Class/LiDAR investigation](../../building_ablation/{paths["building"].name}/extra_occupancy/DIAGNOSIS.md), reuses the 20 absent-building predictions |',
        f'| [5] Optional exploratory input probes | {str(probe_path) if probe_path else "Not supplied for this report"} |',
        '| [6] Published reference | [GaussianFormer3D paper, Table I](https://arxiv.org/html/2505.10685v2), author-reported benchmark; not locally reproduced |',
        '', 'Machine-readable report tables: [ablation_effects.csv](ablation_effects.csv), [condition_metrics.csv](condition_metrics.csv). Source comparison files, checksums and counts are in [study-evidence.json](study-evidence.json); per-seed predictions remain in their original source directories. Figure files are copied beside this report for portable Markdown rendering.',
        '', 'Reproduce the planned studies from `gaussian_parking`:',
        '', '```powershell',
        r'.\evaluation\evaluate_car_appearance_ablation.bat --repeats 20',
        r'.\evaluation\evaluate_building_ablation.bat --repeats 20',
        r'.\evaluation\evaluate_sensor_ablations.bat --baseline-report <realistic-building-present-640-report> --repeats 20',
        '```',
        '', 'Source metadata and manifests record exact capture paths, camera calibration, asset/reference policies, seeds and raw predictions. Run directories use UTC timestamps; their calendar dates can differ from the local preparation date. Repeated executions can differ because model/runtime nondeterminism remains.',
        '', f'Checkpoint SHA-256: `{appearance["checkpoint_sha256"]}`. All 941 tensors match strictly and remain frozen.']
    lines.insert(2, f'Anchor assignment: `{anchor_assignment}` (all comparisons).')
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    evidence={'prepared_local_date':date,'formal_prediction_count':total,'condition_batches':7,'replays_per_condition':20,
        'checkpoint_sha256':appearance['checkpoint_sha256'],
        'comparisons':{k:{'path':str(path/'comparison.json'),'sha256':sha(path/'comparison.json'),
            'validation_path':str(path/'validation.json'),'validation':read(path/'validation.json'),'predictions':len(data[k]['runs'])} for k,path in paths.items()},
        'diagnosis':{'path':str(diagnosis_path/'diagnostics.json'),'sha256':sha(diagnosis_path/'diagnostics.json')},
        'figure_sources':{k:{'path':str(v),'sha256':sha(v)} for k,v in figures.items()},'effect_rows':effect_rows,'condition_rows':condition_rows,
        'report_sha256':sha(output/'REPORT.md')}
    (output/'study-evidence.json').write_text(json.dumps(evidence,indent=2))
    (ROOT/'reports/ablation_study/latest.json').write_text(json.dumps({'report':str(output/'REPORT.md'),'evidence':str(output/'study-evidence.json')},indent=2))
    print('REPORT_READY',output/'REPORT.md',flush=True)

if __name__=='__main__':main()
