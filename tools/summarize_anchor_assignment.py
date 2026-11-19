"""Produce an inspectable comparison from executed anchor-assignment replays."""
import argparse,json
from pathlib import Path
from statistics import mean,median

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('output',type=Path,help='Directory produced by evaluate_anchor_assignment.py')
OUT=parser.parse_args().output.resolve()
data=json.loads((OUT/'results.json').read_text());runs=data['runs']
assert len(runs)==len(data['sources'])*22,'Evaluation is incomplete'
cases=list(data['sources']);summaries={}
for name in cases:
    summaries[name]={}
    for mode in ['random','deterministic']:
        rows=[r for r in runs if r['case']==name and r['mode']==mode and r['repeat']<10]
        same=next(r for r in runs if r['case']==name and r['mode']==mode and r['repeat']==10)
        result={'mean_overlap':mean(r['overlay_jaccard'] for r in rows[1:]),
                'same_seed_overlap':same['overlay_jaccard'],
                'median_forward_ms':1000*median(r['inference_seconds'] for r in rows)}
        if 'raw_score' in rows[0]:
            for policy in ['raw_score','display_score']:
                result[policy]={k:mean(r[policy][k] for r in rows) for k in ['occupied_iou','precision','recall','fp','tp','fn']}
        summaries[name][mode]=result
scored=[n for n in cases if 'raw_score' in summaries[n]['random']]
aggregate={mode:{policy:{k:mean(summaries[n][mode][policy][k] for n in scored)
                       for k in ['occupied_iou','precision','recall']} for policy in ['raw_score','display_score']}
           for mode in ['random','deterministic']}
temporal={}
for mode in ['random','deterministic']:
    values=[r['overlay_jaccard_to_previous'] for r in data['temporal_runs'] if r['mode']==mode and r['overlay_jaccard_to_previous'] is not None]
    temporal[mode]=mean(values) if values else None
summary={'cases':summaries,'equal_condition_accuracy':aggregate,'consecutive_stopped_overlap':temporal,
         'median_forward_ms':{m:1000*median(r['inference_seconds'] for r in runs if r['mode']==m and r['repeat']<10) for m in ['random','deterministic']}}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
lines=['# Deterministic anchor assignment evaluation','',
       'The model now sorts occupied voxels by integer spatial coordinates and uses a private fixed permutation at evaluation time. Global RNG state and voxel emission order no longer choose the assignment. Training retains the original random assignment. Checkpoint tensors and learned fallback anchors are preserved.',
       '', '## Accuracy','',
       f'{len(scored)} scored saved conditions from the supplied comparison. Ten paired seeds per condition/mode, alternating execution order; an extra repeated seed 42 measures residual runtime variation. Ground truth is the pre-existing independent geometry reference. The permutation seed (42), display thresholds and scoring ROI were fixed before evaluation; no tuning was performed.',
       '', '| Metric (equal-condition mean) | Original randomized | Deterministic | Change (pp) |',
       '|---|---:|---:|---:|']
for policy,label in [('raw_score','Raw occupancy'),('display_score','Filtered overlay')]:
    for metric,title in [('occupied_iou','IoU'),('precision','precision'),('recall','recall')]:
        a=aggregate['random'][policy][metric];b=aggregate['deterministic'][policy][metric]
        lines.append(f'| {label} {title} | {a:.2%} | {b:.2%} | {100*(b-a):+.2f} |')
lines+=['','| Condition | Raw IoU: original | Raw IoU: deterministic | Change (pp) |', '|---|---:|---:|---:|']
for n in scored:
    a=summaries[n]['random']['raw_score']['occupied_iou'];b=summaries[n]['deterministic']['raw_score']['occupied_iou']
    lines.append(f'| {n} | {a:.2%} | {b:.2%} | {100*(b-a):+.2f} |')
lines+=['','## Stability','',
        'Overlap is intersection/union of filtered occupied voxel locations before the 3,000-point display cap, using the unchanged >=0.35 confidence, semantic exclusion, and world-height 0.3..2.5 m filters. Scored captures use the fixed above-ground ROI. Different-seed results average seeds 43..51 relative to 42.',
        '', '| Capture | Different-seed overlap: original | Deterministic | Same-seed overlap: original | Deterministic |', '|---|---:|---:|---:|---:|']
for n in cases:
    a=summaries[n]['random'];b=summaries[n]['deterministic']
    lines.append(f"| {n} | {a['mean_overlap']:.1%} | {b['mean_overlap']:.1%} | {a['same_seed_overlap']:.1%} | {b['same_seed_overlap']:.1%} |")
lines+=['',f"Optional fixed-pose temporal overlap (null means no temporal captures supplied): {json.dumps(temporal)}.",
        '', '## Validation and limits','',
        f'- {len(runs)} scored/repeatability forwards plus {len(data["temporal_runs"])} consecutive-capture forwards, in addition to warm-ups.',
        '- CPU and CUDA checks cover empty, sparse, equal and dense inputs; input-order invariance; unique assignment; paired coordinate/intensity writes across two batch samples; unchanged checkpoint parameters; and retained training randomness.',
        '- The 941 checkpoint tensors still load strictly; source input hashes were checked against the existing multiview evidence.',
        '- This does not add missing LiDAR evidence or suppress unsupported learned anchors. Stable errors may remain.',
        '- Four viewpoints in one synthetic lot are not independent scenes. Repeated seeds are replay variation, not additional scene evidence; no general accuracy claim follows.',
        '- The driving start/moving/end captures lack ground truth. Their overlap measures stability only, and the final capture was creeping rather than fully stopped.',
        '- Existing historical evaluation reports describe the old randomized inference policy and are not rewritten.',
        '', '## Reproduce','',
        'From WSL, run `bash tools/evaluate_anchor_assignment.sh --comparison /path/to/comparison.json`, then run `python tools/summarize_anchor_assignment.py /path/to/output`. Sources and hashes, numerical checks, each replay, and representative predictions are preserved beside this file.']
(OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
