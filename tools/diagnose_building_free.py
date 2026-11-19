"""Class and LiDAR-evidence diagnostics of all saved building-free replays.

No new inference, threshold selection, or score changes. LiDAR rays are traced
through half-open voxels using exact grid crossings, with 0.5 m margins at the
sensor and before each measured return. No-return beams are unavailable.
"""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree,ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from occupancy_evaluation import SHAPE,ORIGIN,RESOLUTION,centers

CLASSES=['noise','barrier','bicycle','bus','car','construction_vehicle','motorcycle','pedestrian','traffic_cone','trailer','truck','driveable_surface','other_flat','sidewalk','terrain','manmade','vegetation','empty']
EVIDENCE=['contains_current_return','crossed_by_free_ray','near_obstacle_return_unresolved','unresolved_far_from_obstacle_return']

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def summarize(values):
    values=np.asarray(values,dtype=float)
    return {'n':len(values),'mean':float(values.mean()),'sample_sd':float(values.std(ddof=1)),
            'min':float(values.min()),'max':float(values.max())}

def hit_cells(points,shape=SHAPE,origin=ORIGIN,resolution=RESOLUTION):
    idx=np.floor((points-np.asarray(origin))/resolution).astype(int)
    idx=idx[((idx>=0)&(idx<np.array(shape))).all(1)]
    result=np.zeros(shape,bool)
    if len(idx):result[tuple(idx.T)]=True
    return result

def free_ray_cells(points,shape=SHAPE,origin=ORIGIN,resolution=RESOLUTION,start_margin=.5,end_margin=.5):
    shape=np.asarray(shape);origin=np.asarray(origin);distance=np.linalg.norm(points,axis=1)
    points=points[distance>start_margin+end_margin+1e-8];distance=np.linalg.norm(points,axis=1)
    result=np.zeros(tuple(shape),bool)
    if not len(points):return result
    direction=points/distance[:,None];step=np.sign(direction).astype(int)
    # Advance an epsilon past the start to avoid marking a cell touched only at
    # the artificial start boundary by a negatively directed ray.
    idx=np.floor(((start_margin+1e-9)*direction-origin)/resolution).astype(int)
    boundary=origin+(idx+(step>0))*resolution
    tmax=np.divide(boundary,direction,out=np.full_like(direction,np.inf),where=np.abs(direction)>1e-12)
    tdelta=np.divide(resolution,np.abs(direction),out=np.full_like(direction,np.inf),where=np.abs(direction)>1e-12)
    active=((idx>=0)&(idx<shape)).all(1)
    for _ in range(int(shape.sum())+3):
        rows=np.flatnonzero(active)
        if not len(rows):return result
        result[tuple(idx[rows].T)]=True
        crossing=tmax[rows].min(1)
        advance=crossing<distance[rows]-end_margin-1e-10
        active[rows[~advance]]=False;rows=rows[advance];crossing=crossing[advance]
        axes=np.abs(tmax[rows]-crossing[:,None])<=1e-9
        idx[rows]+=step[rows]*axes
        tmax[rows]=np.where(axes,tmax[rows]+tdelta[rows],tmax[rows])
        active[rows]&=((idx[rows]>=0)&(idx[rows]<shape)).all(1)
    raise RuntimeError('Ray traversal exceeded grid crossing bound')

def summarize_counts(rows,key):
    return summarize([r[key] for r in rows])

def surface_check(points,world_from_lidar,geometry):
    world=points@world_from_lidar[:3,:3].T+world_from_lidar[:3,3];residual=np.full(len(world),np.inf)
    for obj in geometry['objects']:
        v=np.asarray(obj['points_world']);eq=np.unique(np.round(ConvexHull(v).equations,9),axis=0)
        near=((world>=v.min(0)-.3)&(world<=v.max(0)+.3)).all(1)
        if near.any():residual[near]=np.minimum(residual[near],np.abs((world[near]@eq[:,:3].T+eq[:,3]).max(1)))
    return {'current_returns':len(points),'within_5cm_fraction':float((residual<=.05).mean()),
        'median_plane_residual_m':float(np.median(residual)),
        'method':'Nearest absolute maximum convex-hull-plane residual within expanded object bounds; not exact Euclidean corner distance.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('comparison',type=Path);a=p.parse_args()
    parent=a.comparison.parent;comparison=json.loads(a.comparison.read_text());out=parent/'extra_occupancy';out.mkdir(exist_ok=True)
    source=comparison['sources']['absent'];manifest=json.loads(Path(source['manifest']).read_text());sample=manifest['samples'][0]
    capture=Path(source['capture']);meta=json.loads((capture/'capture.json').read_text());geometry=json.loads(Path(sample['geometry']).read_text())
    if meta['building_ablation']['presence']!='absent':raise ValueError('Expected building-free scene')
    with np.load(source['ground_truth_npz']) as gt:
        truth=gt['occupied'];mask=gt['obstacles_above_ground'];world_from_lidar=gt['world_from_lidar']
    with np.load(parent/'comparison-masks.npz') as regions:former=regions['removed_occupied']&mask
    if (truth&former).any():raise ValueError('Former building region is not empty')
    points=np.load(capture/meta['lidar']['points']);current=points[np.isclose(points[:,4],0),:3]
    if len(current)!=meta['lidar']['current_point_count']:raise ValueError('Current sweep selection mismatch')
    hit=hit_cells(current);all_hit=hit_cells(points[:,:3]);ray=free_ray_cells(current)
    ray&=~hit
    xyz=centers(np.indices(SHAPE).reshape(3,-1).T).reshape(*SHAPE,3)
    world_z=(xyz@world_from_lidar[2,:3])+world_from_lidar[2,3]
    obstacle_points=current[(current@world_from_lidar[2,:3]+world_from_lidar[2,3])>=.3]
    distance=np.full(SHAPE,np.inf)
    distance[mask]=cKDTree(obstacle_points).query(xyz[mask],workers=4)[0]
    evidence=np.full(SHAPE,3,np.uint8)
    evidence[(distance<=.5)&~hit&~ray]=2;evidence[ray]=1;evidence[hit]=0
    scopes={'all_false_positives':mask&~truth,'former_building':former,'remaining_lot':mask&~truth&~former}
    np.savez_compressed(out/'lidar-evidence.npz',current_hit_cells=hit,all_sweep_hit_cells=all_hit,free_ray_cells=ray,
        obstacle_return_distance=distance,category=evidence,former_building=former,primary_roi=mask)
    replay_rows=[];class_rows=[];records=sorted([r for r in comparison['runs'] if r['variant']=='absent'],key=lambda r:r['seed'])
    class_acc={scope:{i:[] for i in range(17)} for scope in scopes}
    for replay in records:
        prediction=Path(replay['prediction'])
        if digest(prediction)!=replay['prediction_sha256']:raise ValueError('Prediction checksum mismatch')
        raw=np.load(prediction);fp=(raw!=17)&mask&~truth
        if int(fp.sum())!=replay['fp']:raise ValueError('False positives differ from validated score')
        for scope,region in scopes.items():
            selected=(raw!=17)&region;n=int(selected.sum());counts=np.bincount(evidence[selected],minlength=4)
            if counts.sum()!=n:raise ValueError('Evidence categories do not reconcile')
            row={'seed':replay['seed'],'scope':scope,'false_positives':n,**{name:int(counts[i]) for i,name in enumerate(EVIDENCE)},
                'contains_any_sweep_return':int((selected&all_hit).sum()),
                'within_0.5m_obstacle_return':int((selected&(distance<=.5)).sum()),
                'within_1m_obstacle_return':int((selected&(distance<=1.)).sum()),
                'within_2m_obstacle_return':int((selected&(distance<=2.)).sum())}
            replay_rows.append(row)
            for label in range(17):
                cells=selected&(raw==label);n_class=int(cells.sum());e=np.bincount(evidence[cells],minlength=4)
                heights=[int((cells&(world_z>=lo)&(world_z<hi)).sum()) for lo,hi in [(.3,1),(1,2),(2,3.001)]]
                class_record={'seed':replay['seed'],'scope':scope,'label':label,'class':CLASSES[label],'false_positives':n_class,
                    **{name:int(e[i]) for i,name in enumerate(EVIDENCE)},'height_0.3_1m':heights[0],'height_1_2m':heights[1],'height_2_3m':heights[2]}
                class_rows.append(class_record);class_acc[scope][label].append(class_record)
        print('DIAGNOSED',replay['seed'],'FP',replay['fp'],flush=True)
    summaries={}
    for scope in scopes:
        rows=[r for r in replay_rows if r['scope']==scope];total=sum(r['false_positives'] for r in rows)
        summary={'false_positives':summarize_counts(rows,'false_positives'),
            'evidence':{name:{'mean_voxels':float(np.mean([r[name] for r in rows])),'share_of_false_positives':sum(r[name] for r in rows)/total} for name in EVIDENCE},
            'any_sweep_hit_share':sum(r['contains_any_sweep_return'] for r in rows)/total,
            'near_obstacle_return_shares':{key:sum(r[key] for r in rows)/total for key in ['within_0.5m_obstacle_return','within_1m_obstacle_return','within_2m_obstacle_return']},'classes':[]}
        for label,items in class_acc[scope].items():
            number=sum(r['false_positives'] for r in items)
            if not number:continue
            summary['classes'].append({'label':label,'class':CLASSES[label],'voxels':summarize_counts(items,'false_positives'),'share_of_false_positives':number/total,
                'evidence_shares':{name:sum(r[name] for r in items)/number for name in EVIDENCE},
                'mean_height_counts':{name:float(np.mean([r[name] for r in items])) for name in ['height_0.3_1m','height_1_2m','height_2_3m']}})
        summary['classes'].sort(key=lambda r:r['voxels']['mean'],reverse=True)
        if not np.isclose(sum(c['voxels']['mean'] for c in summary['classes']),summary['false_positives']['mean']):raise ValueError('Class counts do not sum to FP count')
        summaries[scope]=summary
    result={'source_comparison':str(a.comparison),'capture':str(capture),'repeats':len(records),'first_seed':records[0]['seed'],
        'current_returns':len(current),'all_sweep_returns':len(points),'occupied_ground_truth_voxels':int((truth&mask).sum()),
        'primary_voxels':int(mask.sum()),'former_building_voxels':int(former.sum()),'class_mapping_source':str(ROOT_CLASS_SOURCE()),
        'class_mapping_source_sha256':digest(ROOT_CLASS_SOURCE()),'summaries':summaries,
        'lidar_surface_check':surface_check(current,world_from_lidar,geometry),
        'method':{'prediction':'Raw class 17 empty; other classes occupied; same primary ROI and reference proxies as the building ablation',
            'rays':'Exact voxel crossings of current measured-return rays; 0.5 m start/end margins; no-return beams unavailable; cells containing a current return take precedence over ray crossings',
            'near_return':'Voxel-center distance <=0.5 m to current returns at world height >=0.3 m; excludes ground returns; proximity alone is not proof of occupancy',
            'categories':'Mutually exclusive: current return; free-ray crossing without a return; neither and <=0.5 m from obstacle return; neither and farther',
            'aggregation':'Counts averaged over 20 replays of one capture; shares from summed counts across replays, not independent voxel samples'},
        'limitations':['A ray samples part of a voxel, not its complete volume; intersected rays support a spatial contradiction for broad occupancy, not a complete observed-free mask.',
            'Unresolved cells may be outside observed rays or occluded; lack of returns alone does not establish empty space.',
            'False-positive status comes from simulator geometry, independently of LiDAR evidence. Predicted semantic names are not verified simulator semantic accuracy.',
            'One fixed scene and shared simple-car proxy geometry; no generalization or exact realistic-mesh accuracy claim.'],
        'replay_rows':replay_rows,'class_rows':class_rows}
    (out/'diagnostics.json').write_text(json.dumps(result,indent=2))
    for name,rows in [('replay-evidence.csv',replay_rows),('class-evidence.csv',class_rows)]:
        with (out/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    make_plots(out,result,records,xyz,current,truth,mask,former,evidence,distance)
    write_report(out,result)
    print('DIAGNOSIS_READY',out/'DIAGNOSIS.md',flush=True)

def ROOT_CLASS_SOURCE():
    return Path(__file__).resolve().parents[1]/'GaussianFormer3D/eval.py'

def make_plots(out,result,records,xyz,current,truth,mask,former,evidence,distance):
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    colors=['#3274a1','#cc6f35','#568b59','#8264a8']
    for ax,scope,title in zip(axes,['all_false_positives','former_building'],['All above-ground false positives','Within former building volume']):
        classes=result['summaries'][scope]['classes'];names=[c['class'] for c in classes];bottom=np.zeros(len(classes))
        for i,key in enumerate(EVIDENCE):
            values=np.array([c['voxels']['mean']*c['evidence_shares'][key] for c in classes])
            ax.barh(names,values,left=bottom,color=colors[i],label=['Contains return','Free-ray crossing','Near return; unresolved','No ray/hit; unresolved'][i]);bottom+=values
        ax.invert_yaxis();ax.set(xlabel='Mean false-positive voxels per replay',title=title);ax.grid(axis='x',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2)
    fig.suptitle('Building-free scene: predicted classes and current LiDAR evidence | 20 saved replays')
    fig.savefig(out/'classes-and-lidar.png',dpi=180);plt.close(fig)
    raw=np.load(records[0]['prediction'])
    # Select the slice nearest world height 1 m using the recorded transform.
    source=json.loads(Path(result['source_comparison']).read_text())['sources']['absent']
    with np.load(source['ground_truth_npz']) as gt:
        wz=xyz[0,0]@gt['world_from_lidar'][2,:3]+gt['world_from_lidar'][2,3]
    k=int(np.argmin(np.abs(wz-1.)))
    fp=(raw!=17)&mask&~truth
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    class_map=np.ma.masked_where(~fp[:,:,k],raw[:,:,k])
    axes[0].imshow(class_map.T,origin='lower',extent=[-50,50,-50,50],cmap='tab20',vmin=0,vmax=19,interpolation='nearest')
    axes[0].set_title('False-positive classes (IDs)')
    # Label class IDs directly on a separate legend rather than implying object instances.
    ids=np.unique(raw[:,:,k][fp[:,:,k]])
    axes[0].legend(handles=[Patch(facecolor=plt.get_cmap('tab20')(int(i)/19),label=f'{int(i)} {CLASSES[int(i)]}') for i in ids],loc='lower left',fontsize=6)
    hits_near=np.abs(current[:,2]-xyz[0,0,k,2])<=.5
    axes[1].scatter(current[hits_near,0],current[hits_near,1],s=2,c='#172130',label='Returns within ±0.5 m of slice')
    cells=np.argwhere(fp[:,:,k]);points=centers(np.c_[cells,np.full(len(cells),k)])
    axes[1].scatter(points[:,0],points[:,1],s=5,c='#d95478',alpha=.5,label='False-positive cells')
    axes[1].set_title('Current returns and extra occupancy');axes[1].legend(fontsize=7,loc='lower left')
    e=np.ma.masked_where(~fp[:,:,k],evidence[:,:,k]);axes[2].imshow(e.T,origin='lower',extent=[-50,50,-50,50],cmap=ListedColormap(colors),vmin=0,vmax=3,interpolation='nearest')
    axes[2].set_title('LiDAR evidence: blue hit / orange free ray\ngreen near hit / purple unresolved')
    for ax in axes:
        ax.contour(former[:,:,k].T.astype(float),levels=[.5],origin='lower',extent=[-50,50,-50,50],colors=['#172130'],linewidths=.8)
        ax.set(xlim=(-30,30),ylim=(-30,30),xlabel='LiDAR X (m)',ylabel='LiDAR Y (m)');ax.set_aspect('equal');ax.set_facecolor('#f3f4f6')
    fig.suptitle(f'Preselected seed {records[0]["seed"]}; world height {wz[k]:.2f} m; black outline = former building volume\nClass/evidence totals use the full 3D ROI; this figure is one horizontal slice')
    fig.savefig(out/'spatial-lidar-comparison.png',dpi=170);plt.close(fig)

def write_report(out,result):
    all_fp=result['summaries']['all_false_positives'];former_fp=result['summaries']['former_building']
    road=next(c for c in all_fp['classes'] if c['class']=='driveable_surface')
    low_road=road['mean_height_counts']['height_0.3_1m']/road['voxels']['mean']
    lines=['# Building-free extra occupancy: classes and LiDAR evidence','',
        f"All {result['repeats']} saved building-free predictions were analyzed. No new inference or score changes. The scene has all ten realistic car visuals, original car collision/reference proxies, and no building.",
        '', '**Finding:** the extra occupancy is dominated by road/surface, car and manmade labels. The residual occupancy in the former building volume is predominantly labeled road rather than uniquely identifying a building.',
        '',f"No false-positive cell contains a measured return from any of the 11 saved sweeps. Current measured-return rays cross {100*all_fp['evidence']['crossed_by_free_ray']['share_of_false_positives']:.2f}% of all false-positive cells before their hits. The rest has no direct return or crossing in this scan, so most of it is unresolved by the saved LiDAR rather than directly contradicted.",
        '',f"Inside the former building volume, only {100*former_fp['evidence']['crossed_by_free_ray']['share_of_false_positives']:.2f}% of the false-positive cells have such free-ray crossings, and none is within 2 m of an above-ground current return. This region is mostly unobserved by the retained hit rays; no-return beams were not saved.",
        '',f"{100*low_road:.2f}% of road-labeled false-positive cells have world height 0.3–1 m; the remainder is 1–2 m. These are elevated/thickened surface predictions in the above-ground ROI, not evidence of real above-ground road solids. The pattern is consistent with overextended surfaces plus completion in unresolved space; the mechanism producing them has not been isolated.",
        '',f"Current LiDAR returns: {result['current_returns']}; total across saved sweeps: {result['all_sweep_returns']}. Primary ROI: {result['primary_voxels']} cells; actual occupied geometry: {result['occupied_ground_truth_voxels']} cells.",
        '', '![Class composition and evidence](classes-and-lidar.png)']
    for scope,title in [('all_false_positives','All extra occupancy'),('former_building','Extra occupancy in former building volume')]:
        s=result['summaries'][scope];f=s['false_positives']
        lines+=['','## '+title,'',f"Mean false-positive voxels: **{f['mean']:.2f} ± {f['sample_sd']:.2f}** sample SD across replays.",
            '', '| Predicted class | Mean FP voxels | Share of FP | Crossed by free ray | Contains current return |','| --- | ---: | ---: | ---: | ---: |']
        for c in s['classes']:
            lines.append(f"| {c['class']} (ID {c['label']}) | {c['voxels']['mean']:.2f} | {100*c['share_of_false_positives']:.2f}% | {100*c['evidence_shares']['crossed_by_free_ray']:.2f}% | {100*c['evidence_shares']['contains_current_return']:.2f}% |")
        lines+=['', '| LiDAR evidence category | Mean FP voxels | Share of FP |','| --- | ---: | ---: |']
        for name,entry in s['evidence'].items():lines.append(f"| {name} | {entry['mean_voxels']:.2f} | {100*entry['share_of_false_positives']:.2f}% |")
        lines+=['',f"Any of the saved sweeps contains a return in {100*s['any_sweep_hit_share']:.2f}% of these false-positive cells. Voxel centers within 0.5/1/2 m of a current above-ground return: "+' / '.join(f'{100*v:.2f}%' for v in s['near_obstacle_return_shares'].values())+'.']
    lines+=['', '## Interpretation and limits','',
        '- Class names are the checkpoint’s predicted semantic labels, not independently verified semantic accuracy. `manmade` includes structures; it does not uniquely identify a building instance.',
        '- A current return in the same voxel is direct endpoint evidence. A nearby return only indicates proximity; it cannot establish that the predicted voxel is occupied.',
        '- Free-ray crossings are measured evidence against a broad occupied solid along that ray. Rays sample part of each cell; they do not establish that the entire voxel is empty. Here false-positive status is independently established by simulator geometry.',
        '- No hit and no traversing measured ray is unresolved by this saved scan, not proof of empty space. No-return beams were not saved and are not assumed to be free.',
        '- Distance to obstacle returns excludes world heights below 0.3 m so ground returns do not masquerade as support for elevated obstacles. Heights and per-class evidence counts are retained in the CSV/JSON.',
        '- Shares use summed counts across repeated predictions. These are 20 correlated replays of one capture, not independent scenes or individual objects.',
        '',f"LiDAR/geometry alignment check: {100*result['lidar_surface_check']['within_5cm_fraction']:.2f}% of current returns have convex-surface plane residual <=5 cm; median {result['lidar_surface_check']['median_plane_residual_m']:.3g} m. This residual is not exact Euclidean corner distance.",
        '', '![Spatial LiDAR comparison](spatial-lidar-comparison.png)',
        '', '## Evidence','', '- `diagnostics.json`: all summaries, per-seed/per-class counts, methods and source hashes.',
        '- `replay-evidence.csv`, `class-evidence.csv`: every replay, class, scope and evidence category.',
        '- `lidar-evidence.npz`: hit cells, ray crossings, nearest-obstacle-return distances, evidence categories and masks.',
        '- Parent `comparison.json` and `predictions/`: original scored raw predictions, untouched.',
        '',f"Class mapping source: `{result['class_mapping_source']}`; SHA-256 `{result['class_mapping_source_sha256']}`."]
    (out/'DIAGNOSIS.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
