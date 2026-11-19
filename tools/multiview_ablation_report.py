"""Tables and scientific figures for fixed-viewpoint ablation consistency."""
import csv,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Polygon

METRICS=('occupied_iou','precision','recall')
LABELS={'occupied_iou':'Occupied IoU','precision':'Precision','recall':'Recall'}
EFFECTS={'appearance':'Simple → realistic cars','building_whole':'Remove building: whole scene*',
    'building_common':'Remove building: common region','resolution':'640×360 → 960×540 native','history':'11 sweeps → current sweep'}
CONDITIONS={'simple':'Simple cars / building / 640 / 11','reference':'Realistic cars / building / 640 / 11',
    'no_building':'Realistic cars / no building / 640 / 11','high_resolution':'Realistic cars / building / 960 / 11',
    'current_sweep':'Realistic cars / building / 640 / 1'}

def direction(summary,n):
    positive=len(summary['positive_poses']);negative=len(summary['negative_poses'])
    if positive==n:return f'positive at all {n} viewpoints'
    if negative==n:return f'negative at all {n} viewpoints'
    return f'viewpoint-dependent ({positive} positive, {negative} negative)'

def figures(out,result):
    (out/'figures').mkdir(exist_ok=True);poses=result['request']['poses'];colors=['#356aa0','#b95732','#538454','#8b61a5']
    fig,axes=plt.subplots(1,3,figsize=(15,6.8),layout='constrained')
    for ax,key in zip(axes,METRICS):
        for y,effect in enumerate(EFFECTS):
            for i,p in enumerate(poses):
                d=result['per_pose'][p['id']]['effects'][effect][key];mean=100*d['mean'];lo,hi=np.array(d['mean_95pct_t_interval'])*100
                ax.errorbar(mean,y+(i-1.5)*.14,xerr=[[mean-lo],[hi-mean]],fmt='o',markersize=4,color=colors[i],capsize=3,label=p['id'] if y==0 else None)
        ax.axvline(0,color='#555',lw=1);ax.grid(axis='x',alpha=.2)
        ax.set(yticks=range(len(EFFECTS)),yticklabels=list(EFFECTS.values()) if key==METRICS[0] else [],xlabel='Variant − baseline (percentage points)',title=LABELS[key]);ax.invert_yaxis()
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4)
    fig.suptitle('Ablation effects across four fixed viewpoints | 20 matched-seed replays per condition\nWhiskers: within-pose 95% paired t intervals; *building whole-scene targets differ')
    fig.savefig(out/'figures/viewpoint-effects.png',dpi=180);plt.close(fig)
    geo=json.loads(Path(result['sources']['original']['reference']['geometry']).read_text())
    fig,ax=plt.subplots(figsize=(8,7),layout='constrained');ax.add_patch(Rectangle((-21,-15),42,30,facecolor='#e8ebef',edgecolor='#586575'))
    ax.add_patch(Rectangle((-20.8,-4.65),41.6,9.3,facecolor='#f8f9fb',edgecolor='none'))
    for obj in geo['objects']:
        path=obj['path'];points=np.asarray(obj['points_world']);lo=points[:,:2].min(0);hi=points[:,:2].max(0)
        if path.startswith('/World/ParkedVehicles/') and path.endswith('/Body'):
            ax.add_patch(Rectangle(lo,*(hi-lo),facecolor='#8c9ba8',edgecolor='#46525c'))
        elif path=='/World/Building/Main':ax.add_patch(Rectangle(lo,*(hi-lo),facecolor='#d7c5b0',edgecolor='#846c52'));ax.text(0,22,'Building',ha='center')
        elif path.startswith('/World/Cones/') and path.endswith('/Base'):ax.add_patch(Rectangle(lo,*(hi-lo),facecolor='#df8d37'))
    for i,p in enumerate(poses):
        theta=np.radians(p['yaw_deg']);rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        xy=np.array([p['x'],p['y']]);corners=np.array([[-2.2,-1.1],[2.2,-1.1],[2.2,1.1],[-2.2,1.1]])@rotation.T+xy
        ax.add_patch(Polygon(corners,facecolor=colors[i],alpha=.6));ax.arrow(*xy,2.8*np.cos(theta),2.8*np.sin(theta),head_width=.65,color=colors[i],length_includes_head=True)
        ax.text(p['x'],p['y']-2.8,p['id'],ha='center',fontsize=9,color=colors[i])
    ax.set(xlim=(-23,23),ylim=(-17,29),xlabel='World X (m)',ylabel='World Y (m)',title='Fixed viewpoints and requested vehicle footprints\nOne lot; three additional positions/headings');ax.set_aspect('equal');ax.grid(alpha=.15)
    fig.savefig(out/'figures/viewpoint-map.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
    for ax,p in zip(axes.flat,poses):
        capture=Path(result['sources'][p['id']]['reference']['capture']);ax.imshow(Image.open(capture/'CAM_FRONT.png'))
        ax.set_title(f"{p['id']}: ({p['x']:g}, {p['y']:g}) m, yaw {p['yaw_deg']:g}°");ax.axis('off')
    fig.suptitle('Actual front-camera captures | realistic cars, building present, native 640×360')
    fig.savefig(out/'figures/front-camera-viewpoints.png',dpi=170);plt.close(fig)

def build_report(out,result):
    figures(out,result);poses=result['request']['poses'];n=len(poses)
    effect_rows=[];condition_rows=[]
    for p in poses:
        pid=p['id'];d=result['per_pose'][pid]
        for effect,metrics in d['effects'].items():
            for key,s in metrics.items():effect_rows.append({'pose':pid,'x':p['x'],'y':p['y'],'yaw_deg':p['yaw_deg'],
                'effect':effect,'metric':key,'mean_difference_pp':100*s['mean'],'replay_difference_sd_pp':100*s['sample_sd'],
                'interval_low_pp':100*s['mean_95pct_t_interval'][0],'interval_high_pp':100*s['mean_95pct_t_interval'][1],'replays_per_condition':20})
        for variant,metrics in d['conditions'].items():
            for key,s in metrics.items():condition_rows.append({'pose':pid,'condition':variant,'metric':key,'mean_percent':100*s['mean'],
                'replay_sample_sd_pp':100*s['sample_sd'],'min_percent':100*s['min'],'max_percent':100*s['max'],'replays':20})
    for name,rows in [('viewpoint-effects.csv',effect_rows),('viewpoint-condition-metrics.csv',condition_rows)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Fixed-viewpoint ablation consistency','',
        f"{n} position/heading combinations × five conditions × 20 replays = **{len(result['runs'])} new scored predictions**. All four planned ablations are evaluated within each pose. The common-region building score is a secondary scope using the same predictions.",
        '', '## Findings across viewpoints','']
    for effect in ['appearance','building_whole','resolution','history']:
        d=result['across_pose'][effect]['occupied_iou']
        lines.append(f"- **{EFFECTS[effect]}:** IoU effect is {direction(d,n)}. Equal-pose mean {100*d['equal_pose_mean']:+.2f} pp; pose range {100*d['min_pose_effect']:+.2f} to {100*d['max_pose_effect']:+.2f} pp. Within-pose intervals are above zero at {len(d['positive_interval_poses'])}/{n} poses and below zero at {len(d['negative_interval_poses'])}/{n}.")
    lines+=['', '**Important:** the equal-pose means and ranges describe four deliberately selected viewpoints in one lot. The twenty seeds measure inference variation within each viewpoint; they are not additional viewpoints or independent scenes.',
        '', '## Selected viewpoints','', '| Pose | World X (m) | World Y (m) | Heading | Primary scored cells | Occupied cells: building present / absent |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for p in poses:
        c=result['controls'][p['id']];lines.append(f"| {p['id']} | {p['x']:g} | {p['y']:g} | {p['yaw_deg']:g}° | {c['primary_voxels']:,} | {c['occupied_voxels']['reference']:,} / {c['occupied_voxels']['no_building']:,} |")
    lines+=['', '![Viewpoint positions](figures/viewpoint-map.png)','', '![Actual front-camera viewpoints](figures/front-camera-viewpoints.png)',
        '', '## Equal-pose summary of effects','', '| Change | Mean Δ IoU (pp) | Mean Δ precision (pp) | Mean Δ recall (pp) | IoU effect range across poses (pp) | Positive / negative IoU poses |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for effect,label in EFFECTS.items():
        d=result['across_pose'][effect];iou=d['occupied_iou']
        lines.append('| '+label+' | '+' | '.join(f"{100*d[k]['equal_pose_mean']:+.2f}" for k in METRICS)+f" | {100*iou['min_pose_effect']:+.2f} to {100*iou['max_pose_effect']:+.2f} | {len(iou['positive_poses'])} / {len(iou['negative_poses'])} |")
    lines+=['', 'Rows are means of four per-pose paired effects, giving each viewpoint equal weight. They are not voxel-count-weighted pooled IoUs. Building whole-scene targets deliberately change; use the common-region row to assess unchanged geometry.',
        '', '![Per-pose ablation effects and replay intervals](figures/viewpoint-effects.png)',
        '', '## Per-pose paired effects','', 'Each interval is a conditional 95% paired t interval over 20 matching seeds at that pose, not an interval over viewpoints.',
        '', '| Pose | Change | Δ IoU (pp), 95% interval | Δ precision (pp) | Δ recall (pp) |','| --- | --- | ---: | ---: | ---: |']
    for p in poses:
        for effect,label in EFFECTS.items():
            d=result['per_pose'][p['id']]['effects'][effect];iou=d['occupied_iou'];lo,hi=iou['mean_95pct_t_interval']
            lines.append(f"| {p['id']} | {label} | {100*iou['mean']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] | {100*d['precision']['mean']:+.2f} | {100*d['recall']['mean']:+.2f} |")
    lines+=['', '## Absolute primary metrics by pose and condition','', 'Mean ± sample SD across 20 replays; all values in percent. Baseline/reference = ten realistic cars, building present, 640×360 RGB, 11 sweeps.',
        '', '| Pose | Condition | IoU | Precision | Recall |','| --- | --- | ---: | ---: | ---: |']
    for p in poses:
        for variant,label in CONDITIONS.items():
            d=result['per_pose'][p['id']]['conditions'][variant]
            lines.append('| '+p['id']+' | '+label+' | '+' | '.join(f"{100*d[k]['mean']:.2f} ± {100*d[k]['sample_sd']:.2f}" for k in METRICS)+' |')
    lines+=['', '## Controls, interpretation and limits','',
        '- Each new pose is checked against the conservative scene obstacle monitor before capture, settled at the requested position/heading, then held fixed during synchronized RGB/LiDAR/USD capture. There is no driving between evaluation samples.',
        '- The original pose reuses its earlier captured inputs, but all five conditions are rerun under this same unified replay protocol. Small differences from the archived single-pose numbers are expected from runtime nondeterminism and the now-shared reference batch; they are not changes to model weights.',
        '- Within each viewpoint, simple/reference/high-resolution captures must have bit-identical LiDAR point arrays and reference occupancy. Camera extrinsics are matched; high-resolution intrinsics scale by 1.5. All ten detailed visual cars and their original collision/occupancy proxies are checked.',
        '- Removing the building changes camera/LiDAR evidence and its own ground truth. All other geometry must match. The secondary common mask removes exactly the cells that disappear from occupied ground truth and has identical targets between the two scene variants.',
        '- Current-sweep input filters only zero-lag points from that pose’s reference capture. RGB, calibration, image dimensions and current-sweep projected depth stay bit-identical.',
        '- Same frozen checkpoint (941 strict tensors), same raw class-17-empty policy, 0.5 m grid and above-ground ROI rules. The number and composition of target cells vary by viewpoint; comparisons are paired within each pose before equal-pose aggregation.',
        '- Seeds 42–61 are reset within every condition group. Condition order cycles five rotations and their reverses; pose order also rotates/reverses across seeds. Two warm-up forwards per pose/condition are excluded. One model instance runs the measured phase.',
        '- Confidence intervals concern residual seed/runtime variation within a pose only, use a t distribution with 19 degrees of freedom, and are unadjusted for exploratory multiple comparisons. Sign counts and between-pose ranges are descriptive; four selected views in one static lot do not establish generalization to new lots, weather or moving traffic.',
        '- Positions and headings change together in this viewpoint study. It does not separately identify the effect of translation versus camera-rig rotation. All parked cars, cones, lighting and other environmental geometry remain fixed except the explicit appearance/building ablations.',
        '- Shared car proxies still mean this is not exact detailed-mesh occupancy. Full solid and occlusion-inclusive targets are retained. Earlier class/LiDAR diagnostic conclusions apply to the original building-free pose only and are not automatically generalized to these new views.',
        '', '## Evidence','',
        '- `comparison.json`: every prediction, source hash, pose control, per-pose summary and across-pose effect.',
        '- `runs.csv`, `viewpoint-effects.csv`, `viewpoint-condition-metrics.csv`: numerical results.',
        '- `ground-truth/`, `predictions/`: independent reference voxels, common masks and every scored raw prediction.',
        '- `analysis-plan.json`, `comparison-manifest.json`, `capture-runs.json`, capture/export/replay logs: fixed-pose plan and execution record.',
        '- `validation.json`: independent rescoring and consistency checks.',
        '', f"Checkpoint SHA-256: `{result['checkpoint_sha256']}`. No training or model-weight changes."]
    lines.insert(2, f"Anchor assignment: `{result['request'].get('anchor_assignment', 'unrecorded (legacy report)')}`.")
    (out/'VIEWPOINT_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
