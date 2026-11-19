"""Summarize the executed stop/coverage diagnostic, without treating stability as accuracy."""
import json
from pathlib import Path
from statistics import mean

out=Path(__file__).resolve().parents[1]/'reports/stop_prediction_diagnostic'
coverage=json.loads((out/'coverage.json').read_text())
replays=json.loads((out/'replays.json').read_text())
assert len(replays)==40,'Incomplete replay run'
groups={}
for r in replays:groups.setdefault((r['case'],r['variant']),[]).append(r)
summary=[]
for (case,variant),rows in groups.items():
    assert [r['seed'] for r in rows]==[42,42,43,44,45]
    n=rows[0]['actual_lifter_voxels']
    summary.append(dict(case=case,variant=variant,lifter_voxels=n,unassigned_anchors=max(0,25600-n),
                        same_seed_overlap=rows[1]['display_jaccard_to_first'],
                        different_seed_mean_overlap=mean(r['display_jaccard_to_first'] for r in rows[2:]),
                        display_precision_mean=mean(r['display_score']['precision'] for r in rows if r['seed']!=42) if 'display_score' in rows[0] else None))
lines=['# Stop prediction checks','',
       'Source: saved drive coverage and stationary scored capture recorded in the diagnostic inputs.',
       '','40 new model forwards: four captures, full history versus current-only, seeds 42/42/43/44/45. Frozen checkpoint; deployment unchanged.',
       '', '| Capture | Input | Actual LiDAR voxels | Anchors without LiDAR assignment | Same-seed overlap | Different-seed overlap |',
       '|---|---|---:|---:|---:|---:|']
for s in summary:
    lines.append(f"| {s['case']} | {s['variant']} | {s['lifter_voxels']:,} | {s['unassigned_anchors']:,} | {s['same_seed_overlap']:.1%} | {s['different_seed_mean_overlap']:.1%} |")
lines+=['','Overlap is intersection/union of predicted occupied voxel locations after semantic, confidence >=0.35, and world-height 0.3..2.5 m display filters, before the 3,000-point display cap. Different-seed values average seeds 43/44/45 relative to 42. Same-seed is a repeated seed-42 forward; residual differences reveal runtime nondeterminism.',
        '', 'Coverage inventory uses a NumPy voxelization proxy. Table counts are directly measured at the actual model lifter; floating-point boundary differences mean they are not identical.',
        '', '## Interpretation','',
        '- Movement increases unique spatial LiDAR coverage. The stopped startup already has 11 sweeps, but nearly repeated returns.',
        '- Random assignment substantially changes stopped predictions. With the moving capture held fixed, removing historical returns reduces coverage and increases sampling sensitivity.',
        '- Same-seed residual variation means random sampling is not the only source of flicker.',
        '- Driving captures have no independent ground-truth snapshots, so better spatial stability cannot establish better occupancy accuracy.',
        '- The final capture is slow movement (about 0.075 m/s between captures), not a fully stopped endpoint. This run cannot confirm the claimed degradation after a complete final stop.',
        '- Counts of unassigned anchors describe lack of a LiDAR position overwrite, not proof that every such anchor becomes a false positive.',
        '', '## Independently scored stationary capture','']
for s in summary:
    if s['display_precision_mean'] is not None:
        lines.append(f"- {s['variant']}: mean filtered precision {s['display_precision_mean']:.1%} over seeds 43/44/45 in the saved above-ground ground-truth ROI. This uses simplified reference geometry and excludes any claim of real-world accuracy.")
(out/'summary.json').write_text(json.dumps(summary,indent=2))
(out/'CHECKS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
