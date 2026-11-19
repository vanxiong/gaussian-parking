"""Repeated one-factor tests of native image resolution and LiDAR history."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch,cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare_car_appearance import summarize,paired_summary,digest,METRICS,LABELS
from run_gaussianformer3d import ROOT,build_model,build_queries,infer_arrays
from prepare_capture import prepare,prepare_arrays
from occupancy_evaluation import SHAPE,metrics,confusion

VARIANTS=('baseline','high_resolution','current_sweep')
ORDERS=[('baseline','high_resolution','current_sweep'),('high_resolution','current_sweep','baseline'),
        ('current_sweep','baseline','high_resolution'),('baseline','current_sweep','high_resolution'),
        ('current_sweep','high_resolution','baseline'),('high_resolution','baseline','current_sweep')]

def load_inputs(request,out):
    arrays={};sources={};truths={};metas={};geometry={};checks={}
    for variant,expected in [('baseline',[640,360]),('high_resolution',[960,540])]:
        path=Path(request['manifests'][variant]);manifest=json.loads(path.read_text())
        if not manifest.get('stationary_evaluation') or len(manifest['samples'])!=1:raise ValueError('Expected a single stationary capture')
        sample=manifest['samples'][0];capture=Path(sample['capture']);meta=json.loads((capture/'capture.json').read_text())
        appearance=meta['appearance_ablation']
        if appearance['variant']!='realistic' or len(appearance['cars'])!=10 or appearance['selected_car_ids']!=list(range(10)):raise ValueError('Need ten realistic cars')
        if meta['building_ablation']['presence']!='present' or any(c['resolution']!=expected for c in meta['cameras']):raise ValueError('Unexpected scene/resolution')
        if meta['lidar']['sweeps']!=11:raise ValueError('Expected eleven captured sweeps')
        gt_path=path.parent/(capture.name+'-ground-truth.npz')
        with np.load(gt_path) as gt:truths[variant]={k:gt[k].copy() for k in gt.files}
        metas[variant]=meta;arrays[variant]=prepare(capture);geometry[variant]=json.loads(Path(sample['geometry']).read_text())
        sources[variant]={'manifest':str(path),'capture':str(capture),'ground_truth_npz':str(gt_path),
            'scene_sha256':geometry[variant]['scene_sha256'],'lidar_sha256':digest(capture/meta['lidar']['points']),
            'rgb_sha256':{c['name']:digest(capture/c['image']) for c in meta['cameras']},
            'input_rgb_resolution':expected,'lidar_sweeps_used':11,'point_count':len(arrays[variant]['points'])}
    for key in ['world_from_ego','appearance_ablation','building_ablation','lidar']:
        checks[key+'_identical']=metas['baseline'][key]==metas['high_resolution'][key]
    checks['reference_objects_identical']=geometry['baseline']['objects']==geometry['high_resolution']['objects']
    checks['lidar_input_identical_across_resolutions']=np.array_equal(arrays['baseline']['points'],arrays['high_resolution']['points'])
    for key in truths['baseline']:checks['ground_truth_'+key+'_identical']=np.array_equal(truths['baseline'][key],truths['high_resolution'][key])
    cameras={c['name']:c for c in metas['high_resolution']['cameras']}
    calibration=[]
    for c in metas['baseline']['cameras']:
        high=cameras[c['name']];expected=np.asarray(c['intrinsic']).copy();expected[:2]*=1.5
        calibration.append(c['ego_from_optical']==high['ego_from_optical'] and np.allclose(expected,high['intrinsic'],atol=1e-7,rtol=0))
    checks['camera_extrinsics_fixed_intrinsics_scaled']=all(calibration)
    if not all(checks.values()):raise ValueError('Sensor ablation controls failed: '+json.dumps(checks))
    base=sources['baseline'];capture=Path(base['capture']);meta=metas['baseline']
    rgb={c['name']:np.asarray(Image.open(capture/c['image']).convert('RGB')) for c in meta['cameras']}
    points=arrays['baseline']['points'];current=points[np.isclose(points[:,4],0)].copy()
    if len(current)!=meta['lidar']['current_point_count']:raise ValueError('Current-sweep selection mismatch')
    arrays['current_sweep']=prepare_arrays(meta,rgb,current)
    for key in ['img','projection_mat','image_wh','dpt']:
        checks['current_sweep_'+key+'_identical']=np.array_equal(arrays['baseline'][key],arrays['current_sweep'][key])
    if not all(checks[k] for k in checks if k.startswith('current_sweep_')):raise ValueError('Sweep ablation changed nonpoint inputs')
    (out/'inputs').mkdir(exist_ok=True);np.save(out/'inputs/current-lidar.npy',current)
    sources['current_sweep']={**base,'input_transform':'Retain time_lag_seconds == 0 only; original capture still records all 11 sweeps',
        'lidar_sweeps_used':1,'point_count':len(current),'input_points':str(out/'inputs/current-lidar.npy'),
        'input_points_sha256':digest(out/'inputs/current-lidar.npy')}
    checks.update(fixed_world_from_ego=meta['world_from_ego'],point_counts={v:len(arrays[v]['points']) for v in VARIANTS},
        rgb_resolutions={v:sources[v]['input_rgb_resolution'] for v in VARIANTS},
        primary_voxels=int(truths['baseline']['obstacles_above_ground'].sum()))
    return arrays,sources,truths['baseline'],checks

def summarize_result(result):
    rows=result['runs'];ordered={v:sorted([r for r in rows if r['variant']==v],key=lambda r:r['repeat']) for v in VARIANTS}
    result['summary']={v:{k:summarize([r[k] for r in ordered[v]]) for k in METRICS} for v in VARIANTS}
    result['comparisons']={v:{k:paired_summary([r[k] for r in ordered['baseline']],[r[k] for r in ordered[v]]) for k in METRICS} for v in VARIANTS[1:]}
    result['timing_summary']={v:{'median_inference_ms':float(np.median([r['inference_seconds']*1000 for r in ordered[v]])),
        'mean_inference_ms':float(np.mean([r['inference_seconds']*1000 for r in ordered[v]])),
        'peak_cuda_allocated_GB':float(max(r['peak_cuda_allocated_GB'] for r in ordered[v]))} for v in VARIANTS}

def plots(out,result):
    names={'baseline':'640p width / 11 sweeps','high_resolution':'960p width / 11 sweeps','current_sweep':'640p width / 1 sweep'}
    fig,axes=plt.subplots(1,3,figsize=(13,4.7),layout='constrained')
    rows=result['runs'];colors=['#356aa0','#b95732','#538454']
    for ax,key in zip(axes,METRICS):
        for x,v in enumerate(VARIANTS):
            values=np.array([r[key]*100 for r in rows if r['variant']==v])
            ax.scatter(x+np.linspace(-.07,.07,len(values)),values,c=colors[x],s=18)
            ax.errorbar(x,values.mean(),yerr=values.std(ddof=1),fmt='D',color='#172130',capsize=6)
        ax.set(xticks=[0,1,2],xticklabels=['640×360\n11 sweeps','960×540\n11 sweeps','640×360\n1 sweep'],ylabel='Percent',title=LABELS[key],ylim=(0,60),xlim=(-.35,2.35));ax.grid(axis='y',alpha=.2)
    fig.suptitle('Native camera resolution and LiDAR-history ablations | all 10 realistic cars, building present\n20 replays per variant; dots = runs, diamond/whiskers = mean ± sample SD')
    fig.savefig(out/'repeated-metrics.png',dpi=180);plt.close(fig)

def write_report(out,result):
    lines=['# Sensor-input stationary ablations','',
        'One fixed pose with all ten realistic parked cars and the building present. Two factors tested separately: native camera resolution and LiDAR point-history length.',
        '', '## Results','', '20 repeated inferences per condition; mean ± sample SD, in percent. All conditions use identical occupancy ground truth.',
        '', '| Condition | Occupied IoU | Precision | Recall |','| --- | ---: | ---: | ---: |']
    labels={'baseline':'640×360 native RGB, 11 sweeps','high_resolution':'960×540 native RGB, 11 sweeps','current_sweep':'640×360 native RGB, current sweep only'}
    for v in VARIANTS:lines.append('| '+labels[v]+' | '+' | '.join(f"{100*result['summary'][v][k]['mean']:.2f} ± {100*result['summary'][v][k]['sample_sd']:.2f}" for k in METRICS)+' |')
    lines+=['', '| Change from baseline | Metric | Mean difference (pp) | Conditional 95% paired interval (pp) |','| --- | --- | ---: | ---: |']
    for v,comparison in result['comparisons'].items():
        for k,d in comparison.items():
            lo,hi=d['mean_95pct_t_interval'];lines.append(f"| {labels[v]} | {LABELS[k]} | {100*d['mean']:+.2f} | {100*lo:+.2f} to {100*hi:+.2f} |")
    lines+=['','![Repeated metrics](repeated-metrics.png)','', '## Inference cost','',
        '| Condition | Median inference | Peak allocated CUDA memory |','| --- | ---: | ---: |']
    for v,d in result['timing_summary'].items():lines.append(f"| {labels[v]} | {d['median_inference_ms']:.1f} ms | {d['peak_cuda_allocated_GB']:.3f} GB |")
    lines+=['', 'Inference timing is CUDA-synchronized forward plus softmax/argmax, excluding rendering, preprocessing, input transfers and final CPU copy. These are not live capture rates.',
        '', '## Controls and limits','',
        '- Same frozen checkpoint, ego pose, camera extrinsics, building, realistic-car assets/colors/transforms, simulated LiDAR configuration, reference objects, occupancy voxels and scoring masks.',
        '- The 960×540 images were rendered natively from the same cameras. Intrinsics scale by 1.5; images are not enlarged from the 640×360 captures. LiDAR arrays match exactly across resolutions.',
        '- The single-sweep condition filters the exact baseline Nx5 point array to zero time lag. RGB, calibration, image size and current-sweep depth guidance remain bit-identical. This is a saved-input ablation; the source capture still contains eleven sweeps.',
        f"- Actual model input point counts: {result['controls']['point_counts']}. The sweep reduction is genuine, not merely a metadata change.",
        '- All scenes are stationary. Historical sweeps add little viewpoint diversity here; this does not establish how temporal aggregation performs while moving.',
        '- Two warm-up forwards per condition excluded; seeds 42–61 reset per condition; all six execution-order permutations cycled. One baseline prediction per seed is shared by the two contrasts, so the contrasts are not independent experiments.',
        '- Raw class 17 empty, same above-ground primary ROI and shared original car reference geometry. Ground truth is never supplied as model input.',
        '- Conditional paired t intervals concern seed/runtime variation at one pose only. Tests are exploratory; no multiplicity correction is applied and no generalization claim is made.',
        '', '## Evidence','',
        '- `comparison.json`, `runs.csv`: every prediction, confusion count, summary, pair difference and timing.',
        '- `predictions/`: raw model output for every seed and condition.',
        '- `inputs/current-lidar.npy`: exact current-only point input; original capture unchanged.',
        '- `controls.json`, `comparison-manifest.json`, `analysis-plan.json`, `validation.json`: controls, planned comparisons, source manifests and independent verification.']
    lines.insert(2, f"Anchor assignment: `{result['request'].get('anchor_assignment', 'unrecorded (legacy report)')}`.")
    (out/'COMPARISON.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);a=p.parse_args();request=json.loads(a.manifest.read_text());out=a.manifest.parent
    arrays,sources,truth,checks=load_inputs(request,out);(out/'controls.json').write_text(json.dumps(checks,indent=2));print('CONTROLS_PASSED',json.dumps(checks),flush=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.manual_seed(request['first_seed']);np.random.seed(request['first_seed']);model,config,tensors=build_model(anchor_assignment=request['anchor_assignment']);queries=build_queries(config)
    if tensors!=941 or any(p.requires_grad for p in model.parameters()):raise RuntimeError('Expected unchanged frozen checkpoint')
    for v in VARIANTS:
        for _ in range(request['warmup_per_variant']):infer_arrays(model,config,arrays[v],queries)
    (out/'predictions').mkdir(exist_ok=True);rows=[]
    result={'request':request,'sources':sources,'controls':checks,'execution_orders':[list(o) for o in ORDERS],
        'strict_tensor_count':tensors,'checkpoint_sha256':digest(ROOT/'checkpoints/surroundocc_release.pth'),
        'device':torch.cuda.get_device_name(),'pytorch':torch.__version__,'ground_truth_used_as_model_input':False,'runs':rows}
    for repeat in range(request['repeats_per_variant']):
        seed=request['first_seed']+repeat
        for position,v in enumerate(ORDERS[repeat%len(ORDERS)]):
            torch.manual_seed(seed);np.random.seed(seed);raw,_,timing=infer_arrays(model,config,arrays[v],queries)
            if raw.shape!=SHAPE or raw.max()>17:raise ValueError('Invalid prediction')
            path=out/'predictions'/f'{v}-seed-{seed}.npy';np.save(path,raw);score=metrics(confusion(raw!=17,truth['occupied'],truth['obstacles_above_ground']))
            rows.append({'variant':v,'repeat':repeat,'seed':seed,'order_in_pair':position,'prediction':str(path),'prediction_sha256':digest(path),**score,**timing})
            (out/'partial-runs.json').write_text(json.dumps(rows,indent=2));print('REPLAY',repeat+1,v,seed,score['occupied_iou'],flush=True)
    summarize_result(result);(out/'comparison.json').write_text(json.dumps(result,indent=2))
    keys=['variant','repeat','seed','order_in_pair','tp','fp','fn','tn',*METRICS,'prediction','prediction_sha256','inference_seconds']
    with (out/'runs.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    plots(out,result);write_report(out,result);print('COMPARISON_READY',out/'COMPARISON.md',flush=True)

if __name__=='__main__':main()
