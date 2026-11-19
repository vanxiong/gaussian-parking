"""Rerun the archived stationary ablation baseline with both anchor policies."""
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean,stdev

ROOT=Path(__file__).resolve().parents[1]

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.repeats<2:parser.error('At least two repeats required')
    if not args.comparison.is_file():parser.error('--comparison does not exist')
    source=args.comparison.resolve()
    archived=json.loads(source.read_text())
    out=(args.output or ROOT/'reports/anchor_assignment_baseline'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')).resolve();out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch,cv2
    from prepare_capture import prepare
    from run_gaussianformer3d import build_model,build_queries,infer_arrays
    from occupancy_evaluation import confusion,metrics
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.manual_seed(42);np.random.seed(42)
    model,config,tensors=build_model();queries=build_queries(config)
    assert tensors==941 and not any(p.requires_grad for p in model.parameters())
    checkpoint=digest(ROOT/'checkpoints/surroundocc_release.pth')
    assert checkpoint==archived['checkpoint_sha256']
    result={'archive':str(source),'archive_sha256':digest(source),'checkpoint_sha256':checkpoint,
            'assignment_helper_sha256':digest(ROOT/'GaussianFormer3D/model/lifter/anchor_assignment.py'),
            'configuration':'Realistic parked cars, building present, 640x360 cameras, 11 LiDAR sweeps; frozen weights',
            'metric':'Raw argmax class !=17, unchanged obstacles_above_ground mask, no confidence/semantic display filtering',
            'repeats':args.repeats,'first_seed':args.seed,'anchor_policies':['random','deterministic'],'sources':{},'runs':[]}
    for pose,sources in archived['sources'].items():
        source_record=sources['reference'];capture=Path(source_record['capture']);truthpath=Path(source_record['ground_truth_npz'])
        metadata=json.loads((capture/'capture.json').read_text())
        assert digest(capture/'capture.json')==source_record['capture_sha256']
        assert digest(capture/metadata['lidar']['points'])==source_record['lidar_sha256']
        for camera in metadata['cameras']:
            assert digest(capture/camera['image'])==source_record['rgb_sha256'][camera['name']]
            assert camera['resolution']==[640,360]
        assert metadata['lidar']['sweeps']==11
        assert metadata['appearance_ablation']['variant']=='realistic'
        assert metadata['building_ablation']['presence']=='present'
        gt=np.load(truthpath);truth=gt['occupied'];mask=gt['obstacles_above_ground']
        assert int(mask.sum())==archived['controls'][pose]['primary_voxels']
        assert int((truth&mask).sum())==archived['controls'][pose]['occupied_voxels']['reference']
        arrays=prepare(capture)
        result['sources'][pose]={**source_record,'ground_truth_sha256':digest(truthpath)}
        for mode in ['random','deterministic']:
            model.lifter.deterministic_eval=mode=='deterministic'
            for _ in range(2):infer_arrays(model,config,arrays,queries)
        for repeat in range(args.repeats):
            seed=args.seed+repeat
            for mode in (['random','deterministic'] if repeat%2==0 else ['deterministic','random']):
                model.lifter.deterministic_eval=mode=='deterministic'
                torch.manual_seed(seed);np.random.seed(seed)
                raw,confidence,timing=infer_arrays(model,config,arrays,queries)
                score=metrics(confusion(raw!=17,truth,mask))
                result['runs'].append({'pose':pose,'mode':mode,'repeat':repeat,'seed':seed,**score,**timing})
                if repeat==0:np.savez_compressed(out/(pose+'-'+mode+'.npz'),raw=raw,confidence=confidence)
            (out/'results.json').write_text(json.dumps(result,indent=2))
        print('BASELINE_POSE_COMPLETE',pose,flush=True)
    model.lifter.deterministic_eval=True
    summary={}
    for pose in archived['sources']:
        summary[pose]={'archived':archived['per_pose'][pose]['conditions']['reference']}
        for mode in ['random','deterministic']:
            rows=[r for r in result['runs'] if r['pose']==pose and r['mode']==mode]
            summary[pose][mode]={k:{'mean':mean(r[k] for r in rows),'sample_sd':stdev(r[k] for r in rows)}
                                 for k in ['occupied_iou','precision','recall']}
    aggregate={m:{k:mean(s[m][k]['mean'] for s in summary.values()) for k in ['occupied_iou','precision','recall']}
               for m in ['archived','random','deterministic']}
    result['summary']=summary;result['equal_pose_means']=aggregate
    (out/'results.json').write_text(json.dumps(result,indent=2))
    lines=['# Stationary ablation baseline after anchor assignment fix','',
           'Same four saved baseline captures and independent occupancy targets as the archived multiview ablations: realistic cars, building present, 640x360 RGB, eleven LiDAR sweeps. Captures were stationary. Replays per pose/policy and seed range are recorded in results.json, alternating randomized/deterministic order, two warm-ups per policy. All 941 frozen checkpoint tensors match. Input hashes and target ROI/occupied counts were verified.',
           '', 'Primary score: raw argmax occupancy (class 17 empty), same above-ground ROI; no visualization filtering or threshold tuning.',
           '', '| Viewpoint | Archived IoU | Fresh randomized IoU | Deterministic IoU | Change vs fresh (pp) |', '|---|---:|---:|---:|---:|']
    for pose,s in summary.items():
        a=s['archived']['occupied_iou']['mean'];b=s['random']['occupied_iou']['mean'];c=s['deterministic']['occupied_iou']['mean']
        lines.append(f'| {pose} | {a:.2%} | {b:.2%} | {c:.2%} | {100*(c-b):+.2f} |')
    a,b,c=[aggregate[m]['occupied_iou'] for m in ['archived','random','deterministic']]
    lines.append(f'| **Equal-viewpoint mean** | **{a:.2%}** | **{b:.2%}** | **{c:.2%}** | **{100*(c-b):+.2f}** |')
    lines+=['', '| Equal-viewpoint metric | Fresh randomized | Deterministic | Change (pp) |','|---|---:|---:|---:|']
    for k in ['occupied_iou','precision','recall']:
        b=aggregate['random'][k];c=aggregate['deterministic'][k]
        lines.append(f'| {k} | {b:.2%} | {c:.2%} | {100*(c-b):+.2f} |')
    lines+=['', 'The fresh randomized comparison isolates the assignment change better than comparing only with the archived score, because residual runtime nondeterminism affects replay scores. Repeated seeds are not independent viewpoints. These four selected viewpoints in one synthetic lot do not establish generalization.',
            '', 'This reruns the baseline only. Old ablation differences remain historical: measuring whether appearance, building removal, resolution or sweep history has the same effect under the new policy requires rerunning both members of each ablation. A stationary car does not itself invalidate the earlier tests.',
            '', 'Reproduce in WSL: `bash tools/evaluate_anchor_baseline.sh --comparison /path/to/comparison.json`. Per-replay scores, source hashes, target hashes, and representative predictions are stored beside this file.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    assert len(result['runs'])==len(archived['sources'])*2*args.repeats
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
