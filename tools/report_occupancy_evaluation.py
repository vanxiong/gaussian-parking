"""Write a compact results report and scientific comparison figure."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);args=p.parse_args();output=args.manifest.parent
    manifest=json.loads(args.manifest.read_text());accuracy=json.loads((output/'accuracy.json').read_text());latency=json.loads((output/'latency.json').read_text())
    metadata=[json.loads((Path(s['capture'])/'capture.json').read_text()) for s in manifest['samples']]
    resolutions=sorted({tuple(c['resolution']) for m in metadata for c in m['cameras']})
    sweeps=[m['lidar']['sweeps'] for m in metadata]
    latency['inputs']={'source_rgb_resolutions':resolutions,'camera_count':6,'lidar_sweeps':sweeps,'precision':'FP32'}
    live_result=json.loads((Path(manifest['run'])/'result.json').read_text())
    updates=live_result['perception_updates'];live=[r['latency_wall_seconds'] for r in updates]
    def distribution(values):
        return {'n':len(values),'mean_seconds':float(np.mean(values)),'median_seconds':float(np.median(values)),
                'p95_seconds':float(np.percentile(values,95)),'min_seconds':float(min(values)),'max_seconds':float(max(values))}
    if len(live)>2:
        latency['live_capture_to_result_all']=distribution(live)
        latency['live_capture_to_result_warm']=distribution(live[2:])
        latency['live_capture_to_result_samples_seconds']=live
        latency['scopes']['live_capture_to_result_warm']='Instrumented live simulator: synchronized sensor snapshot start to model response accepted by main loop; includes rendering, ground-truth snapshot export, disk I/O, worker inference and scheduling; excludes first 2 updates and model startup'
    (output/'latency.json').write_text(json.dumps(latency,indent=2))
    primary='obstacles_above_ground/argmax';score=accuracy['aggregate'][primary]
    first=manifest['samples'][0];capture=Path(first['capture']);gt=np.load(output/(capture.name+'-ground-truth.npz'))
    raw=np.load(first['raw_labels']);mask=gt['obstacles_above_ground'];truth=gt['occupied'];prediction=raw!=17
    world_from_lidar=gt['world_from_lidar'];local_z=1.-world_from_lidar[2,3]
    k=int(np.clip(np.round((local_z+5)/.5-.5),0,15))
    slice_mask=mask[:,:,k];t=truth[:,:,k];pred=prediction[:,:,k]
    errors=np.zeros(t.shape,int);errors[t&pred]=1;errors[~t&pred]=2;errors[t&~pred]=3
    cmap=ListedColormap(['#f2f3f5','#228854','#d94e84','#eaaa35']);cmap.set_bad('#ccd1d8')
    binary_cmap=ListedColormap(['#f2f3f5','#254d86']);binary_cmap.set_bad('#ccd1d8')
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    for ax,image,title in zip(axes,[t,pred,errors],['USD solid occupancy','GaussianFormer3D argmax','TP green / FP pink / FN amber']):
        shown=np.ma.masked_where(~slice_mask,image)
        ax.imshow(shown.T,origin='lower',extent=[-50,50,-50,50],cmap=cmap if title.startswith('TP') else binary_cmap,vmin=0,vmax=3 if title.startswith('TP') else 1,interpolation='nearest')
        ax.set(xlim=(-30,30),ylim=(-30,30),xlabel='LiDAR X (m)',ylabel='LiDAR Y (m)',title=title);ax.set_aspect('equal')
    fig.suptitle(f'Paired occupancy slice near world Z = 1 m; gray area excluded | {capture.name}')
    fig.savefig(output/'occupancy-comparison.png',dpi=180);plt.close(fig)
    pct=lambda value:'undefined' if value is None else f'{100*value:.2f}%'
    lines=['# GaussianFormer3D — Isaac Sim occupancy evaluation','',
        f"{len(manifest['samples'])} synchronized poses from one parking scene; frozen weights; {latency['strict_tensor_count']} strict checkpoint tensors.",'',
        '## Primary: above-ground occupied geometry','',
        '| Metric | Result |','| --- | ---: |']
    for key in ['occupied_iou','precision','recall','f1','voxel_accuracy','all_empty_baseline_accuracy']:lines.append(f'| {key} | {pct(score[key])} |')
    lines+=['',f"Confusion totals: TP {score['tp']}, FP {score['fp']}, FN {score['fn']}, TN {score['tn']}. Counts are summed across correlated frames.",'',
        'Voxel accuracy is dominated by free space; occupied IoU, precision and recall are the main scores. The all-empty baseline has zero occupied recall and zero occupied IoU when objects exist.','',
        '## Latency','',f"Device: {latency['device']}. {latency['warmup_forwards']} warm-up forwards, then {latency['timed_forwards']} timed batch-1 forwards ({manifest['repeats']} repetitions per pose). Startup {latency['startup_seconds']:.2f} s is separate.",
        '', f"Six RGB cameras, source resolution(s) {resolutions} (padded during preprocessing); LiDAR sweeps {min(sweeps)}..{max(sweeps)}; FP32 model inference. The simulator is closed during replay timing.",'',
        '| Scope | Median | P95 | Mean |','| --- | ---: | ---: | ---: |']
    for scope in ['inference','preprocessing','pipeline']+(['live_capture_to_result_warm'] if len(live)>2 else []):
        row=latency[scope];lines.append(f"| {scope} | {row['median_seconds']*1000:.1f} ms | {row['p95_seconds']*1000:.1f} ms | {row['mean_seconds']*1000:.1f} ms |")
    lines+=['',f"Peak allocated CUDA memory: {latency['peak_cuda_allocated_GB']:.3f} GB (allocator peak, not total GPU utilization).",'',
        *[f'- **{key}:** {value}' for key,value in latency['scopes'].items()],
        '', '## Protocol and limitations','',
        *[f'- **{key}:** {value}' for key,value in accuracy['protocol'].items()],
        *[f'- {value}' for value in accuracy['limitations']],
        '', 'Semantic mIoU is not measured. These geometric scores are not comparable to published nuScenes/SurroundOcc scores. No thresholds were selected using these results.','',
        '## Files','',
        '- `accuracy.json`: confusion counts and metrics for each frame, including the ground-inclusive and confidence-filtered secondary policies.',
        '- `latency.json`: every timing sample, scope, warm-up, checkpoint hash and hardware.',
        '- `manifest.json`: exact inputs, per-frame ground truth and prediction paths.',
        '- `*-ground-truth.npz`: occupied voxels, evaluation masks and world transform.',
        '- `*-geometry.json`: per-capture USD geometry and scene hash.','',
        '![Independent ground truth and model prediction](occupancy-comparison.png)','']
    (output/'REPORT.md').write_text('\n'.join(lines))
    print('REPORT_READY',str(output/'REPORT.md'),flush=True)

if __name__=='__main__':main()
