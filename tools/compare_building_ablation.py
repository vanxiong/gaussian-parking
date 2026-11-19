"""Paired repeated building-presence ablation with per-scene and common targets."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from compare_car_appearance import digest,summarize,paired_summary,METRICS,LABELS
from run_gaussianformer3d import ROOT,build_model,build_queries,infer_arrays
from prepare_capture import prepare
from occupancy_evaluation import SHAPE,confusion,metrics

VARIANTS=('present','absent')

def load_inputs(request,out):
    frames={};arrays={};truths={};metadata={};geometry={};checks={}
    for variant in VARIANTS:
        path=Path(request['manifests'][variant]);manifest=json.loads(path.read_text())
        if not manifest.get('stationary_evaluation') or len(manifest['samples'])!=1:raise ValueError('Need one stationary capture per variant')
        sample=manifest['samples'][0];capture=Path(sample['capture']);meta=json.loads((capture/'capture.json').read_text())
        appearance=meta['appearance_ablation'];building=meta['building_ablation']
        if appearance['variant']!='realistic' or appearance['selected_car_ids']!=list(range(10)) or len(appearance['cars'])!=10:raise ValueError('All ten cars must be realistic')
        if building['presence']!=variant:raise ValueError('Incorrect recorded building presence')
        geo=json.loads(Path(sample['geometry']).read_text())
        if len(geo['appearance_reference']['excluded_detailed_visual_paths'])!=730:raise ValueError('Detailed visual meshes missing')
        gt_path=path.parent/(capture.name+'-ground-truth.npz')
        with np.load(gt_path) as gt:truths[variant]={k:gt[k].copy() for k in gt.files}
        metadata[variant]=meta;geometry[variant]=geo;arrays[variant]=prepare(capture)
        frames[variant]={'manifest':str(path),'capture':str(capture),'ground_truth_npz':str(gt_path),
            'ground_truth_sha256':digest(gt_path),'scene_sha256':geo['scene_sha256'],
            'rgb_sha256':{c['name']:digest(capture/c['image']) for c in meta['cameras']},
            'lidar_sha256':digest(capture/meta['lidar']['points']),
            'building_ablation':building,'appearance_ablation':appearance}
    for key in ['world_from_ego','cameras','appearance_ablation']:
        checks[key+'_identical']=metadata['present'][key]==metadata['absent'][key]
    for key in ['ego_from_lidar','fields','sweeps','simulation_timestamp_seconds','scan_model','intensity_model']:
        checks['lidar_'+key+'_identical']=metadata['present']['lidar'][key]==metadata['absent']['lidar'][key]
    for key in ['projection_mat','image_wh']:
        checks['prepared_'+key+'_identical']=np.array_equal(arrays['present'][key],arrays['absent'][key])
    for key in ['obstacles_above_ground','including_ground','world_from_lidar']:
        checks['ground_truth_'+key+'_identical']=np.array_equal(truths['present'][key],truths['absent'][key])
    def objects(variant,building):
        return {o['path']:o for o in geometry[variant]['objects'] if o['path'].startswith('/World/Building/')==building}
    removed=objects('present',True)
    checks['only_building_geometry_changed']=objects('present',False)==objects('absent',False)
    checks['building_removed_completely']=bool(removed) and not objects('absent',True)
    checks['known_xy_bounds_identical']=geometry['present']['known_xy_bounds']==geometry['absent']['known_xy_bounds']
    checks['absence_adds_no_occupied_cells']=not np.any(truths['absent']['occupied']&~truths['present']['occupied'])
    if not all(checks.values()):raise ValueError('Uncontrolled scene comparison: '+json.dumps(checks))
    primary=truths['present']['obstacles_above_ground']
    removed_cells=truths['present']['occupied']&~truths['absent']['occupied']
    common=primary&~removed_cells
    if not np.array_equal(truths['present']['occupied'][common],truths['absent']['occupied'][common]):raise ValueError('Common-region truth differs')
    if not np.any(removed_cells&primary):raise ValueError('Building removal did not change primary ground truth')
    checks.update(common_region_truth_identical=True,removed_geometry_paths=list(removed),
        removed_primary_occupied_voxels=int((removed_cells&primary).sum()),primary_voxels=int(primary.sum()),
        common_region_voxels=int(common.sum()),ground_truth_occupied_voxels={v:int((truths[v]['occupied']&primary).sum()) for v in VARIANTS},
        lidar_points_shapes={v:list(arrays[v]['points'].shape) for v in VARIANTS},
        changed_rgb_files=[name for name in frames['present']['rgb_sha256'] if frames['present']['rgb_sha256'][name]!=frames['absent']['rgb_sha256'][name]],
        lidar_inputs_changed=not np.array_equal(arrays['present']['points'],arrays['absent']['points']),
        fixed_world_from_ego=metadata['present']['world_from_ego'])
    np.savez_compressed(out/'comparison-masks.npz',primary=primary,common_region=common,removed_occupied=removed_cells)
    return frames,arrays,truths,common,checks

def summarize_scope(rows,scope):
    def value(row,key):return row[key] if scope=='primary' else row['common_region'][key]
    ordered={v:sorted([r for r in rows if r['variant']==v],key=lambda r:r['repeat']) for v in VARIANTS}
    summary={v:{k:summarize([value(r,k) for r in ordered[v]]) for k in METRICS} for v in VARIANTS}
    paired={k:paired_summary([value(r,k) for r in ordered['present']],[value(r,k) for r in ordered['absent']]) for k in METRICS}
    return summary,paired

def plots(out,result,truths,common):
    rows=result['runs'];request=result['request'];colors=['#356aa0','#b95732']
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for index,scope in enumerate(['primary','common_region']):
        for ax,key in zip(axes[index],METRICS):
            values={v:np.array([r[key] if scope=='primary' else r['common_region'][key] for r in rows if r['variant']==v])*100 for v in VARIANTS}
            for left,right in zip(values['present'],values['absent']):ax.plot([0,1],[left,right],color='#b0b6bf',alpha=.45,lw=.7,zorder=1)
            for x,v in enumerate(VARIANTS):
                a=values[v];ax.scatter(x+np.linspace(-.06,.06,len(a)),a,c=colors[x],s=16,zorder=2)
                ax.errorbar(x,a.mean(),yerr=a.std(ddof=1),fmt='D',color='#172130',capsize=6,markersize=5,zorder=3)
            scope_title='Whole scene (different targets)' if scope=='primary' else 'Common region (identical targets)'
            ax.set(xticks=[0,1],xticklabels=['Building','No building'],ylabel='Percent',title=scope_title+'\n'+LABELS[key],xlim=(-.3,1.3),ylim=(0,100))
            ax.grid(axis='y',alpha=.2)
    fig.suptitle(f'All 10 cars realistic | {request["repeats_per_variant"]} paired replays per variant\nDots: runs; diamonds/whiskers: mean ± sample SD; lines: matching seeds')
    fig.savefig(out/'repeated-metrics.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained');seed=request['first_seed']
    cmap=ListedColormap(['#f2f3f5','#228854','#d94e84','#eaaa35']);cmap.set_bad('#ccd1d8')
    binary=ListedColormap(['#f2f3f5','#254d86']);binary.set_bad('#ccd1d8')
    for index,v in enumerate(VARIANTS):
        gt=truths[v];k=int(np.clip(np.round((1.-gt['world_from_lidar'][2,3]+5)/.5-.5),0,15))
        t=gt['occupied'][:,:,k];p=np.load(out/'predictions'/f'{v}-seed-{seed}.npy')[:,:,k]!=17;mask=gt['obstacles_above_ground'][:,:,k]
        error=np.zeros(t.shape,int);error[t&p]=1;error[~t&p]=2;error[t&~p]=3
        for ax,im,title in zip(axes[index],[t,p,error],[f'Building {v}: ground truth',f'Building {v}: prediction','TP green / FP pink / FN amber']):
            ax.imshow(np.ma.masked_where(~mask,im).T,origin='lower',extent=[-50,50,-50,50],cmap=cmap if title.startswith('TP') else binary,vmin=0,vmax=3 if title.startswith('TP') else 1,interpolation='nearest')
            ax.set(xlim=(-30,30),ylim=(-30,30),xlabel='LiDAR X (m)',ylabel='LiDAR Y (m)',title=title);ax.set_aspect('equal')
    fig.suptitle(f'First preselected seed {seed}; horizontal slice near world Z=1 m\nBuilding removed from visuals, collision/LiDAR and ground truth; car proxies unchanged')
    fig.savefig(out/'paired-occupancy-comparison.png',dpi=160);plt.close(fig)

def write_report(out,result):
    request=result['request'];controls=result['controls'];n=request['repeats_per_variant']
    lines=['# Building-presence stationary ablation','',
        f'All 10 parked cars use realistic visual models in both variants. {n} paired repeated inferences per variant at the same fixed pose; the driving car is unchanged.',
        '', '**Important:** removing the building changes the images, LiDAR and occupancy target. Whole-scene scores compare two different target populations; they do not isolate improved prediction on the same objects.',
        '',f"Primary evaluated cells: {controls['primary_voxels']}. Occupied target: building present **{controls['ground_truth_occupied_voxels']['present']}**, absent **{controls['ground_truth_occupied_voxels']['absent']}**. Removed occupied cells: **{controls['removed_primary_occupied_voxels']}**."]
    for scope,title in [('primary','Whole-scene results (own ground truth per variant)'),('common_region','Common-region results (identical ground truth)')]:
        summary=result['summary'] if scope=='primary' else result['common_region_summary']
        paired=result['paired_differences'] if scope=='primary' else result['common_region_paired_differences']
        lines+=['','## '+title,'','Values are mean ± sample SD over repeated inference; differences use unrounded values.',
            '', '| Metric | Building present (%) | Building absent (%) | Absent − present (pp) | 95% paired mean-difference interval (pp) |',
            '| --- | ---: | ---: | ---: | ---: |']
        for key in METRICS:
            a=summary['present'][key];b=summary['absent'][key];d=paired[key];lo,hi=d['mean_95pct_t_interval']
            lines.append(f"| {LABELS[key]} | {100*a['mean']:.2f} ± {100*a['sample_sd']:.2f} | {100*b['mean']:.2f} ± {100*b['sample_sd']:.2f} | {100*d['mean']:+.2f} | {100*lo:+.2f} to {100*hi:+.2f} |")
        if scope=='common_region':lines+=['',f"The common mask excludes the {controls['removed_primary_occupied_voxels']} formerly occupied building cells from both scores and retains {controls['common_region_voxels']} cells. All other evaluated voxels have identical truth. This mask was defined before inference from geometry, not selected to improve scores. The former building volume is still evaluated as empty in the building-absent whole-scene score."]
    former_volume_fp=np.mean([r['fp']-r['common_region']['fp'] for r in result['runs'] if r['variant']=='absent'])
    lines+=['','## Interpretation','',
        'Whole-scene and common-region results answer different questions. The large whole-scene recall increase comes from removing a difficult occupied target; recall on the unchanged nonbuilding geometry is the common-region recall and must be assessed separately.',
        '',f'The building-absent predictions still occupy an average of {former_volume_fp:.2f} cells inside the now-empty former building volume. Those are false positives in the whole-scene score and are excluded from both common-region scores. This helps explain why the two IoU comparisons can move in opposite directions.',
        '', 'These effects apply to this fixed pose and scene, not to general performance across parking lots.',
        '','![Repeated metrics](repeated-metrics.png)','', '## Controls and protocol','',
        '- Default stationary pose: world XY (-5,-2) m, yaw 0 degrees. Six 640×360 cameras; 11 LiDAR sweeps. Same above-ground ROI and raw argmax empty class 17. No threshold tuning.',
        '- Exact equality checks passed for ego pose, camera calibration, realistic car transforms/colors/asset hashes, LiDAR sensor configuration, ground footprint, ROI masks and all nonbuilding reference objects.',
        f"- Only `/World/Building` was removed ({len(controls['removed_geometry_paths'])} shapes including windows and canopy); visuals, collision and ground truth removed together. LiDAR arrays deliberately differ: {controls['lidar_points_shapes']}. Changed RGB files: {controls['changed_rgb_files']}.",
        '- Both variants retain the same simple-car collision and occupancy proxies beneath realistic visual meshes. Building removal affects actual sensor evidence; it is not an appearance-only building test.',
        f"- Seeds {request['first_seed']}–{request['first_seed']+n-1}, matching within pairs; execution order alternates. Two warm-up forwards per variant are excluded. Frozen checkpoint, {result['strict_tensor_count']} strict tensors; SHA-256 `{result['checkpoint_sha256']}`.",
        '- The paired t intervals quantify seed/runtime variation at one pose, assuming paired differences are sufficiently independent. They are not uncertainty across scenes, layouts or vehicles. Residual runtime nondeterminism remains despite seed resets.',
        '- These are repeated model replays of one synchronized capture per variant, not independent simulator scenes. Ground truth is never supplied as model input.',
        '', '## Representative occupancy','', 'The first preselected seed is shown, not the best run. The visualization is a slice; metrics use the full 3D ROI.',
        '', '![Paired occupancy](paired-occupancy-comparison.png)','', '## Evidence','',
        '- `comparison.json`, `runs.csv`: every inference, confusion count, whole-scene/common-region metric and summary.',
        '- `predictions/`: all raw predictions, with checksums in the JSON.',
        '- `comparison-masks.npz`: primary/common masks and removed occupied cells.',
        '- `controls.json`, `analysis-plan.json`, `comparison-manifest.json`: checks, predeclared protocol and source manifests.',
        '- `validation.json`: independent recomputation from saved predictions.',
        '', 'Source reports:', '']
    for v,source in result['sources'].items():lines.append(f"- {v}: `{source['manifest']}`")
    lines.insert(2, f"Anchor assignment: `{result['request'].get('anchor_assignment', 'unrecorded (legacy report)')}`.")
    (out/'COMPARISON.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('--report-only',action='store_true');a=p.parse_args();request=json.loads(a.manifest.read_text());out=a.manifest.parent
    if a.report_only:
        write_report(out,json.loads((out/'comparison.json').read_text()));return
    if request['repeats_per_variant']<2:raise ValueError('At least two paired replays required')
    sources,arrays,truths,common,controls=load_inputs(request,out)
    (out/'controls.json').write_text(json.dumps(controls,indent=2));print('INPUT_CONTROLS_PASSED',json.dumps(controls),flush=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.manual_seed(request['first_seed']);np.random.seed(request['first_seed']);model,config,tensors=build_model(anchor_assignment=request['anchor_assignment']);queries=build_queries(config)
    if tensors!=941 or any(p.requires_grad for p in model.parameters()):raise RuntimeError('Expected frozen strict checkpoint')
    for v in VARIANTS:
        for _ in range(request['warmup_per_variant']):infer_arrays(model,config,arrays[v],queries)
    (out/'predictions').mkdir(exist_ok=True);rows=[]
    result={'request':request,'sources':sources,'controls':controls,'strict_tensor_count':tensors,'device':torch.cuda.get_device_name(),
        'checkpoint_sha256':digest(ROOT/'checkpoints/surroundocc_release.pth'),'pytorch':torch.__version__,'ground_truth_used_as_model_input':False,'runs':rows}
    for repeat in range(request['repeats_per_variant']):
        seed=request['first_seed']+repeat
        for position,v in enumerate(VARIANTS if repeat%2==0 else VARIANTS[::-1]):
            torch.manual_seed(seed);np.random.seed(seed);raw,_,timing=infer_arrays(model,config,arrays[v],queries)
            if raw.shape!=SHAPE or raw.max()>17:raise ValueError('Invalid raw model output')
            path=out/'predictions'/f'{v}-seed-{seed}.npy';np.save(path,raw);gt=truths[v]
            score=metrics(confusion(raw!=17,gt['occupied'],gt['obstacles_above_ground']))
            secondary=metrics(confusion(raw!=17,gt['occupied'],common))
            if any(score[k] is None or secondary[k] is None for k in METRICS):raise ValueError('Undefined metric')
            rows.append({'variant':v,'repeat':repeat,'seed':seed,'order_in_pair':position,'prediction':str(path),'prediction_sha256':digest(path),**score,'common_region':secondary,**timing})
            (out/'partial-runs.json').write_text(json.dumps(rows,indent=2))
            print('REPLAY',repeat+1,v,seed,'IoU',score['occupied_iou'],'common IoU',secondary['occupied_iou'],flush=True)
    result['summary'],result['paired_differences']=summarize_scope(rows,'primary')
    result['common_region_summary'],result['common_region_paired_differences']=summarize_scope(rows,'common_region')
    (out/'comparison.json').write_text(json.dumps(result,indent=2))
    keys=['variant','repeat','seed','order_in_pair','tp','fp','fn','tn',*METRICS,*['common_'+k for k in ['tp','fp','fn','tn',*METRICS]],'prediction','prediction_sha256']
    with (out/'runs.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore');writer.writeheader()
        for r in rows:writer.writerow({**r,**{'common_'+k:v for k,v in r['common_region'].items()}})
    plots(out,result,truths,common);write_report(out,result);print('COMPARISON_READY',out/'COMPARISON.md',flush=True)

if __name__=='__main__':main()
