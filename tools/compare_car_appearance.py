"""Repeated paired comparison of stationary simple and realistic car appearances.

Saved predictions and per-run confusion counts make every reported value
recomputable. The experiment measures inference variation at one fixed pose.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch,cv2
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from run_gaussianformer3d import ROOT,build_model,build_queries,infer_arrays
from prepare_capture import prepare
from occupancy_evaluation import SHAPE,confusion,metrics

METRICS=('occupied_iou','precision','recall')
LABELS={'occupied_iou':'Occupied IoU','precision':'Precision','recall':'Recall'}
VARIANTS=('simple','realistic')

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def summarize(values):
    v=np.asarray(values,dtype=float)
    if len(v)<2 or not np.isfinite(v).all():raise ValueError('Need finite repeated observations')
    return {'n':len(v),'mean':float(v.mean()),'sample_sd':float(v.std(ddof=1)),
            'min':float(v.min()),'max':float(v.max())}

def paired_summary(simple,realistic):
    differences=np.asarray(realistic)-np.asarray(simple)
    result=summarize(differences)
    half=float(stats.t.ppf(.975,len(differences)-1)*result['sample_sd']/np.sqrt(len(differences)))
    return {**result,'mean_95pct_t_interval':[result['mean']-half,result['mean']+half],
            'positive_pairs':int((differences>0).sum()),'negative_pairs':int((differences<0).sum()),
            'interval_scope':'Repeated seed/runtime variation at this fixed pose only; not scene-generalization uncertainty'}

def load_inputs(request):
    frames={};arrays={};truths={};checks={}
    for variant in VARIANTS:
        path=Path(request['manifests'][variant]);manifest=json.loads(path.read_text())
        if not manifest.get('stationary_evaluation') or len(manifest['samples'])!=1:raise ValueError('Expected one stationary capture per variant')
        sample=manifest['samples'][0];capture=Path(sample['capture']);metadata=json.loads((capture/'capture.json').read_text())
        ablation=metadata['appearance_ablation']
        if ablation['variant']!=variant:raise ValueError('Incorrect variant metadata')
        if ablation['selected_car_ids']!=list(range(10)):raise ValueError('Expected all 10 parked cars in experiment metadata')
        if variant=='realistic' and len(ablation['cars'])!=10:raise ValueError('Realistic capture did not replace all 10 cars')
        geometry=json.loads(Path(sample['geometry']).read_text())
        if variant=='realistic' and len(geometry['appearance_reference']['excluded_detailed_visual_paths'])!=730:raise ValueError('Expected 730 detailed visual meshes')
        gt_path=path.parent/(capture.name+'-ground-truth.npz')
        with np.load(gt_path) as gt:truths[variant]={k:gt[k].copy() for k in gt.files}
        arrays[variant]=prepare(capture)
        frames[variant]={'manifest':str(path),'capture':str(capture),'metadata':metadata,'geometry':geometry,
            'ground_truth_npz':str(gt_path),'ground_truth_sha256':digest(gt_path),
            'rgb_sha256':{c['name']:digest(capture/c['image']) for c in metadata['cameras']},
            'lidar_sha256':digest(capture/metadata['lidar']['points']),
            'appearance_ablation':ablation}
    for key in ['world_from_ego','cameras','lidar']:
        checks[key+'_identical']=frames['simple']['metadata'][key]==frames['realistic']['metadata'][key]
    checks['geometry_objects_identical']=frames['simple']['geometry']['objects']==frames['realistic']['geometry']['objects']
    for key in ['points','projection_mat','image_wh','dpt']:
        checks['prepared_'+key+'_identical']=np.array_equal(arrays['simple'][key],arrays['realistic'][key])
    for key in truths['simple']:
        checks['ground_truth_'+key+'_identical']=np.array_equal(truths['simple'][key],truths['realistic'][key])
    if not all(checks.values()):raise ValueError('Comparison is not controlled: '+json.dumps(checks))
    if np.array_equal(arrays['simple']['img'],arrays['realistic']['img']):raise ValueError('Expected changed camera images')
    checks['different_rgb_arrays']=True
    checks['changed_rgb_files']=[name for name in frames['simple']['rgb_sha256'] if frames['simple']['rgb_sha256'][name]!=frames['realistic']['rgb_sha256'][name]]
    checks['lidar_points_shape']=list(arrays['simple']['points'].shape)
    checks['fixed_world_from_ego']=frames['simple']['metadata']['world_from_ego']
    checks['camera_resolutions']=[c['resolution'] for c in frames['simple']['metadata']['cameras']]
    checks['lidar_sweeps']=frames['simple']['metadata']['lidar']['sweeps']
    for frame in frames.values():
        del frame['metadata'];del frame['geometry']
    return frames,arrays,truths['simple'],checks

def plots(out,rows,truth,request):
    colors={'simple':'#356aa0','realistic':'#b95732'}
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout='constrained')
    for ax,key in zip(axes,METRICS):
        values={name:np.array([r[key] for r in rows if r['variant']==name])*100 for name in VARIANTS}
        for i,(left,right) in enumerate(zip(values['simple'],values['realistic'])):
            ax.plot([0,1],[left,right],color='#b0b6bf',alpha=.5,lw=.7,zorder=1)
        for x,name in enumerate(VARIANTS):
            v=values[name];jitter=np.linspace(-.06,.06,len(v))
            ax.scatter(x+jitter,v,c=colors[name],s=20,label=name,zorder=2)
            ax.errorbar(x,v.mean(),yerr=v.std(ddof=1),fmt='D',color='#172130',capsize=7,markersize=6,zorder=3)
        ax.set(xticks=[0,1],xticklabels=['Simple','10 realistic'],ylabel='Percent',title=LABELS[key],xlim=(-.3,1.3),ylim=(0,50))
        ax.grid(axis='y',alpha=.2)
    fig.suptitle(f'{request["repeats_per_variant"]} paired inference runs per variant | one fixed pose\nDots: individual runs; diamonds/whiskers: mean ± sample SD; lines: matching seeds')
    fig.savefig(out/'repeated-metrics.png',dpi=180);plt.close(fig)
    # Representative map is the first preselected seed, never the best scoring run.
    k=int(np.clip(np.round((1.-truth['world_from_lidar'][2,3]+5)/.5-.5),0,15))
    mask=truth['obstacles_above_ground'][:,:,k];t=truth['occupied'][:,:,k]
    cmap=ListedColormap(['#f2f3f5','#228854','#d94e84','#eaaa35']);cmap.set_bad('#ccd1d8')
    binary=ListedColormap(['#f2f3f5','#254d86']);binary.set_bad('#ccd1d8')
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    seed=request['first_seed']
    for index,name in enumerate(VARIANTS):
        raw=np.load(out/'predictions'/f'{name}-seed-{seed}.npy');p=raw[:,:,k]!=17
        errors=np.zeros(t.shape,int);errors[t&p]=1;errors[~t&p]=2;errors[t&~p]=3
        for ax,im,title in zip(axes[index],[t,p,errors],['Shared ground-truth occupancy',name+' prediction','TP green / FP pink / FN amber']):
            ax.imshow(np.ma.masked_where(~mask,im).T,origin='lower',extent=[-50,50,-50,50],cmap=cmap if title.startswith('TP') else binary,vmin=0,vmax=3 if title.startswith('TP') else 1,interpolation='nearest')
            ax.set(xlim=(-30,30),ylim=(-30,30),xlabel='LiDAR X (m)',ylabel='LiDAR Y (m)',title=title);ax.set_aspect('equal')
    fig.suptitle(f'Preselected seed {seed}; slice near world height 1 m\nMetrics use the full 3D ROI; gray excluded; detailed cars use shared reference proxies')
    fig.savefig(out/'paired-occupancy-comparison.png',dpi=160);plt.close(fig)

def write_report(out,result):
    summaries=result['summary'];paired=result['paired_differences'];n=result['request']['repeats_per_variant']
    lines=['# All-ten-car stationary appearance ablation','',
        f'{n} paired repeated inferences per variant at one fixed parking-lot pose. All ten parked cars switch appearance; the driving car is unchanged.',
        '', '**Metric definition:** above-ground occupied IoU, precision and recall using raw argmax class 17 as empty. No threshold tuning. Same original simple-car occupancy-reference solids and LiDAR colliders in both variants.',
        '', '| Metric | Simple mean ± SD | Realistic mean ± SD | Mean difference (percentage points) |','| --- | ---: | ---: | ---: |']
    for key in METRICS:
        s=summaries['simple'][key];r=summaries['realistic'][key];d=paired[key]
        lines.append(f"| {LABELS[key]} | {100*s['mean']:.2f}% ± {100*s['sample_sd']:.2f} pp | {100*r['mean']:.2f}% ± {100*r['sample_sd']:.2f} pp | {100*d['mean']:+.2f} |")
    lines+=['', '![Repeated metrics](repeated-metrics.png)','',
        '| Metric | Simple min–max | Realistic min–max | 95% interval for mean paired difference (pp) |','| --- | ---: | ---: | ---: |']
    for key in METRICS:
        s=summaries['simple'][key];r=summaries['realistic'][key];lo,hi=paired[key]['mean_95pct_t_interval']
        lines.append(f"| {LABELS[key]} | {100*s['min']:.2f}–{100*s['max']:.2f}% | {100*r['min']:.2f}–{100*r['max']:.2f}% | {100*lo:+.2f} to {100*hi:+.2f} |")
    lines+=['', '## Interpretation','']
    for key in METRICS:
        d=paired[key];lo,hi=d['mean_95pct_t_interval'];direction='higher' if d['mean']>0 else 'lower'
        evidence='The interval spans zero, so the direction is unresolved against measured inference variation.' if lo<=0<=hi else 'The paired interval excludes zero for this fixed capture; this does not establish a benefit across scenes.'
        lines.append(f"{LABELS[key]} is {abs(100*d['mean']):.2f} percentage points {direction} on average with realistic visuals. {evidence}")
        lines.append('')
    lines+=['## Controls and scope','',
        f"- Synchronized stationary captures of both variants; camera resolutions {result['controls']['camera_resolutions']}, LiDAR sweeps {result['controls']['lidar_sweeps']}.",
        '- Exact equality checks passed for ego pose, camera/LiDAR metadata, prepared LiDAR points, depth guidance, projection matrices, image dimensions, exported reference objects, occupied voxels and evaluation masks.',
        f"- All 10 detailed cars are verified in capture metadata; 730 visual meshes are excluded from scoring and the original car solids are retained. {len(result['controls']['changed_rgb_files'])} / 6 RGB files changed.",
        f"- Frozen checkpoint: {result['strict_tensor_count']} tensors matched strictly; SHA-256 `{result['checkpoint_sha256']}`. Device: {result['device']}.",
        f"- Two warm-up forwards per variant are excluded. Seeds {result['request']['first_seed']} through {result['request']['first_seed']+n-1} are reset for both members of each pair; execution order alternates. One shared model instance is used. Ground truth is never supplied to the model.",
        '- SD is run-to-run spread, not uncertainty across scenes. The 95% paired t intervals concern inference variation at this single pose and assume paired run differences are sufficiently independent. Runtime nondeterminism can remain despite seed resets.',
        '- These are repeated inferences of two saved sensor captures, not 40 independent scenes or repeated simulator startups. Shared proxy targets isolate appearance sensitivity; they do not measure exact detailed-mesh reconstruction.',
        '- The filled, occlusion-inclusive building remains in both variants and still dominates much of the occupancy target. No new region or class exclusions were selected after looking at results.',
        '', '## Representative predictions','',
        'The following uses the first preselected seed, not the best performing run. The plot is a horizontal slice; the scores use the complete primary 3D ROI.',
        '', '![Paired occupancy](paired-occupancy-comparison.png)',
        '', '## Evidence','',
        '- `comparison.json`: all controls, per-run confusion counts, means, SDs, ranges and paired differences.',
        '- `runs.csv`: one row per model inference.',
        '- `predictions/`: raw labels for every variant/seed; enough to independently recompute scores.',
        '- `analysis-plan.json`, `comparison-manifest.json`: protocol and exact source report manifests.',
        '- `simple-capture-evaluation.log`, `realistic-capture-evaluation.log`, `replay.log`: execution records.',
        '', 'Source evaluation directories:', '']
    for name,frame in result['sources'].items():lines.append(f"- {name}: `{frame['manifest']}`")
    lines.insert(2, f"Anchor assignment: `{result['request'].get('anchor_assignment', 'unrecorded (legacy report)')}`.")
    (out/'COMPARISON.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);a=p.parse_args()
    request=json.loads(a.manifest.read_text());out=a.manifest.parent
    if request['repeats_per_variant']<2:raise ValueError('Need repeated inference')
    sources,arrays,truth,controls=load_inputs(request)
    (out/'controls.json').write_text(json.dumps(controls,indent=2))
    print('INPUT_CONTROLS_PASSED',json.dumps(controls),flush=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.manual_seed(request['first_seed']);np.random.seed(request['first_seed'])
    model,config,tensors=build_model(anchor_assignment=request['anchor_assignment']);queries=build_queries(config)
    if tensors!=941 or any(p.requires_grad for p in model.parameters()):raise RuntimeError('Expected unchanged frozen checkpoint')
    for name in VARIANTS:
        for _ in range(request['warmup_per_variant']):infer_arrays(model,config,arrays[name],queries)
    directory=out/'predictions';directory.mkdir(exist_ok=True);rows=[]
    result={'request':request,'sources':sources,'controls':controls,'strict_tensor_count':tensors,
        'checkpoint_sha256':digest(ROOT/'checkpoints/surroundocc_release.pth'),'device':torch.cuda.get_device_name(),
        'pytorch':torch.__version__,'ground_truth_used_as_model_input':False,'runs':rows}
    for repeat in range(request['repeats_per_variant']):
        seed=request['first_seed']+repeat;order=VARIANTS if repeat%2==0 else VARIANTS[::-1]
        for position,name in enumerate(order):
            torch.manual_seed(seed);np.random.seed(seed)
            raw,_,timing=infer_arrays(model,config,arrays[name],queries)
            if raw.shape!=SHAPE or raw.max()>17:raise ValueError('Invalid model labels')
            prediction=directory/f'{name}-seed-{seed}.npy';np.save(prediction,raw)
            scored=metrics(confusion(raw!=17,truth['occupied'],truth['obstacles_above_ground']))
            if any(scored[k] is None for k in METRICS):raise ValueError('Undefined primary metric')
            rows.append({'repeat':repeat,'seed':seed,'variant':name,'order_in_pair':position,
                'prediction':str(prediction),'prediction_sha256':digest(prediction),**scored,**timing})
            (out/'partial-runs.json').write_text(json.dumps(rows,indent=2))
            print('REPLAY',repeat+1,name,seed,' '.join(f'{k}={100*scored[k]:.3f}%' for k in METRICS),flush=True)
    by_variant={name:sorted([r for r in rows if r['variant']==name],key=lambda r:r['repeat']) for name in VARIANTS}
    result['summary']={name:{k:summarize([r[k] for r in by_variant[name]]) for k in METRICS} for name in VARIANTS}
    result['paired_differences']={k:paired_summary([r[k] for r in by_variant['simple']],[r[k] for r in by_variant['realistic']]) for k in METRICS}
    (out/'comparison.json').write_text(json.dumps(result,indent=2))
    keys=['repeat','seed','variant','order_in_pair','tp','fp','fn','tn',*METRICS,'f1','evaluated_voxels','prediction','prediction_sha256','inference_seconds']
    with (out/'runs.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    plots(out,rows,truth,request);write_report(out,result)
    print('COMPARISON_READY',str(out/'COMPARISON.md'),flush=True)

if __name__=='__main__':main()
