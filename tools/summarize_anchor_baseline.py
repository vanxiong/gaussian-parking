"""Independently rescore saved predictions before summarizing the baseline test."""
import csv,json
from pathlib import Path
import numpy as np
from scipy import stats
from evaluate_anchor_baseline import digest
from occupancy_evaluation import confusion,metrics

KEYS=['occupied_iou','precision','recall']


def summarize(out):
    data=json.loads((out/'results.json').read_text());runs=data['runs'];n=data['repeats_per_policy']
    assert len(runs)==len(data['sources'])*2*n
    cases={}
    for name,source in data['sources'].items():
        gt_path=Path(source['ground_truth_npz']);assert digest(gt_path)==source['ground_truth_sha256']
        gt=np.load(gt_path)
        for r in [r for r in runs if r['case']==name]:
            path=Path(r['prediction']);assert digest(path)==r['prediction_sha256']
            raw=np.load(path);recomputed=metrics(confusion(raw!=17,gt['occupied'],gt['obstacles_above_ground']))
            assert all(r[k]==v for k,v in recomputed.items()),'Saved score mismatch'
        records={m:sorted([r for r in runs if r['case']==name and r['mode']==m],key=lambda r:r['seed']) for m in ['random','deterministic']}
        assert all([r['seed'] for r in records[m]]==list(range(42,42+n)) for m in records)
        old=[r for r in data['archived_runs'] if r['case']==name]
        summary={'archived':{},'random':{},'deterministic':{},'paired_effect':{}}
        for k in KEYS:
            for m,rows in [('archived',old),*records.items()]:
                v=np.array([r[k] for r in rows]);summary[m][k]={'mean':float(v.mean()),'sample_sd':float(v.std(ddof=1)),'n':len(v)}
            delta=np.array([b[k]-a[k] for a,b in zip(records['random'],records['deterministic'])])
            half=float(stats.t.ppf(.975,n-1)*delta.std(ddof=1)/np.sqrt(n))
            summary['paired_effect'][k]={'mean':float(delta.mean()),'conditional_95pct_interval':[float(delta.mean()-half),float(delta.mean()+half)]}
        cases[name]=summary
    baseline=[name for name in cases if name.endswith('_reference')]
    equal_pose={mode:{k:float(np.mean([cases[name][mode][k]['mean'] for name in baseline])) for k in KEYS} for mode in ['archived','random','deterministic']}
    result={'cases':cases,'four_viewpoint_reference_equal_pose_mean':equal_pose,
            'validation':{'passed':True,'independently_rescored_predictions':len(runs),'ground_truth_hashes_checked':len(cases)},
            'interval_scope':'Conditional seed/runtime replay variation at each fixed pose, not viewpoint or scene uncertainty.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    fields=['case','mode','seed','tp','fp','fn','tn',*KEYS,'prediction','prediction_sha256']
    with (out/'runs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(runs)
    lines=['# Stationary ablation baseline: assignment fix','',
        'The archived ablation captures were stationary. This test replays the exact shared reference condition: ten realistic parked cars, building present, six 640×360 cameras, eleven LiDAR sweeps, frozen checkpoint. Four fixed viewpoints are measured with 20 paired replays per assignment policy. The original simple-car condition is included as a supplemental baseline.',
        '', 'Primary occupied IoU uses raw argmax: class 17 is empty, all others occupied, with the identical archived above-ground ROI. No confidence, semantic-display, or point-count filters affect this score.',
        '', '| Baseline | Archived IoU | Fresh original assignment | Updated assignment | Paired change (pp) | Conditional 95% interval (pp) |', '|---|---:|---:|---:|---:|---:|']
    for name,s in cases.items():
        effect=s['paired_effect']['occupied_iou'];lo,hi=effect['conditional_95pct_interval']
        lines.append(f"| {name} | {s['archived']['occupied_iou']['mean']:.2%} | {s['random']['occupied_iou']['mean']:.2%} | {s['deterministic']['occupied_iou']['mean']:.2%} | {100*effect['mean']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines+=['','## Four-viewpoint reference mean','', '| Metric | Archived | Fresh original | Updated | Paired change (pp) |','|---|---:|---:|---:|---:|']
    for k in KEYS:
        a,b,c=(equal_pose[m][k] for m in ['archived','random','deterministic'])
        lines.append(f'| {k} | {a:.2%} | {b:.2%} | {c:.2%} | {100*(c-b):+.2f} |')
    lines+=['','## Controls and scope','',
        f'- {len(runs)} saved predictions independently rescored; all hashes, confusion counts and reported metrics matched.',
        '- Checkpoint hash matches the historical study; all 941 tensors load strictly. Capture metadata, six RGB files and LiDAR hashes match archived evidence. Camera size, sweep count, pose, ROI size and occupied target counts are checked.',
        '- Seeds 42..61 match the prior study; policies alternate execution order within each seed. Two warm-up forwards per policy/capture precede measurement. The fixed inference permutation was selected before this test and was not tuned.',
        '- The fresh original-assignment run is the primary comparator. Differences from archived values reflect replay/runtime variation; historical reports are preserved.',
        '- Equal-pose means weight each of the four reference viewpoints equally. Supplemental simple-car scores are excluded from that mean.',
        '- Per-pose intervals quantify repeated seed/runtime variation only. Four selected viewpoints in one synthetic lot do not establish generalization; deterministic assignment can preserve incorrect predictions.',
        '', '## Reproduce','', '`bash tools/evaluate_anchor_baseline.sh` from WSL. Saved inputs and targets are reused, so no simulator movement or recapture is involved.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    import sys
    summarize(Path(sys.argv[1]))
