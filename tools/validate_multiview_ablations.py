"""Independently rescore all viewpoint predictions and verify aggregation."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
from validate_car_appearance_comparison import local

CONTRASTS={'appearance':('simple','reference'),'building_whole':('reference','no_building'),
    'building_common':('reference','no_building'),'resolution':('reference','high_resolution'),'history':('reference','current_sweep')}
METRICS=('occupied_iou','precision','recall')

def close(a,b):
    if not math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-10):raise AssertionError(f'Numeric mismatch: {a} vs {b}')

def main():
    p=argparse.ArgumentParser();p.add_argument('comparison',type=Path);a=p.parse_args();out=a.comparison.parent;d=json.loads(a.comparison.read_text())
    poses=[p['id'] for p in d['request']['poses']];variants=list(d['sources'][poses[0]]);n=d['request']['repeats_per_condition'];rows=d['runs']
    if n!=20 or len(rows)!=len(poses)*len(variants)*n:raise AssertionError('Unexpected study population')
    gt={};masks={}
    for pid in poses:
        if not all(d['controls'][pid]['checks'].values()):raise AssertionError('Failed input control')
        gt[pid]={}
        for variant,source in d['sources'][pid].items():
            with np.load(local(source['ground_truth_npz'])) as saved:gt[pid][variant]={k:saved[k].copy() for k in saved.files}
        primary=gt[pid]['reference']['obstacles_above_ground'];removed=gt[pid]['reference']['occupied']&~gt[pid]['no_building']['occupied']
        masks[pid]=primary&~removed
        with np.load(local(d['controls'][pid]['common_mask'])) as saved:
            if not np.array_equal(saved['common_region'],masks[pid]):raise AssertionError('Incorrect common mask')
        for v in ['simple','reference','high_resolution','current_sweep']:
            if not np.array_equal(gt[pid][v]['occupied'],gt[pid]['reference']['occupied']):raise AssertionError('Nonbuilding truth changed')
        if not np.array_equal(gt[pid]['reference']['occupied'][masks[pid]],gt[pid]['no_building']['occupied'][masks[pid]]):raise AssertionError('Common truth differs')
    for row in rows:
        path=local(row['prediction'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['prediction_sha256']:raise AssertionError('Prediction checksum mismatch')
        raw=np.load(path);truth=gt[row['pose']][row['variant']]
        for scope,mask in [('primary',truth['obstacles_above_ground']),('common_region',masks[row['pose']])]:
            p=raw[mask]!=17;t=truth['occupied'][mask]
            tp=int((p&t).sum());fp=int((p&~t).sum());fn=int((~p&t).sum());tn=int((~p&~t).sum())
            expected={'tp':tp,'fp':fp,'fn':fn,'tn':tn,'occupied_iou':tp/(tp+fp+fn),'precision':tp/(tp+fp),'recall':tp/(tp+fn)}
            recorded=row if scope=='primary' else row[scope]
            for key,value in expected.items():close(value,recorded[key])
    intervals=0
    for pid in poses:
        ordered={v:sorted([r for r in rows if r['pose']==pid and r['variant']==v],key=lambda r:r['seed']) for v in variants}
        for v,records in ordered.items():
            if len(records)!=n or [r['seed'] for r in records]!=list(range(d['request']['first_seed'],d['request']['first_seed']+n)):raise AssertionError('Seed coverage differs')
            for key in METRICS:
                values=[r[key] for r in records];mean=math.fsum(values)/n;sd=math.sqrt(math.fsum((v-mean)**2 for v in values)/(n-1))
                recorded=d['per_pose'][pid]['conditions'][v][key]
                for name,value in [('mean',mean),('sample_sd',sd),('min',min(values)),('max',max(values))]:close(value,recorded[name])
        for repeat in range(n):
            group=sorted([r for r in rows if r['pose']==pid and r['repeat']==repeat],key=lambda r:r['order_in_condition_group'])
            if [r['variant'] for r in group]!=d['variant_orders'][repeat%len(d['variant_orders'])]:raise AssertionError('Condition execution order differs')
        for contrast,(first,second) in CONTRASTS.items():
            for key in METRICS:
                diffs=[b['common_region'][key]-a['common_region'][key] if contrast=='building_common' else b[key]-a[key] for a,b in zip(ordered[first],ordered[second])]
                mean=math.fsum(diffs)/n;sd=math.sqrt(math.fsum((x-mean)**2 for x in diffs)/(n-1));half=2.093024054408263*sd/math.sqrt(n)
                recorded=d['per_pose'][pid]['effects'][contrast][key]
                close(mean,recorded['mean']);close(sd,recorded['sample_sd']);close(mean-half,recorded['mean_95pct_t_interval'][0]);close(mean+half,recorded['mean_95pct_t_interval'][1]);intervals+=1
    for contrast in CONTRASTS:
        for key in METRICS:
            values=[d['per_pose'][pid]['effects'][contrast][key]['mean'] for pid in poses]
            mean=math.fsum(values)/len(poses);sd=math.sqrt(math.fsum((v-mean)**2 for v in values)/(len(poses)-1));recorded=d['across_pose'][contrast][key]
            for name,value in [('equal_pose_mean',mean),('between_pose_sample_sd',sd),('min_pose_effect',min(values)),('max_pose_effect',max(values))]:close(value,recorded[name])
            if recorded['positive_poses']!=[pid for pid,v in zip(poses,values) if v>0] or recorded['negative_poses']!=[pid for pid,v in zip(poses,values) if v<0]:raise AssertionError('Pose sign counts differ')
    with (out/'runs.csv').open(newline='',encoding='utf-8') as stream:exported=list(csv.DictReader(stream))
    if len(exported)!=len(rows):raise AssertionError('CSV row count differs')
    for csv_row,row in zip(exported,rows):
        for key in ['tp','fp','fn','tn',*METRICS]:close(csv_row[key],row[key]);close(csv_row['common_'+key],row['common_region'][key])
    validation={'passed':True,'poses':len(poses),'conditions_per_pose':len(variants),'replays_per_condition':n,
        'predictions_recomputed':len(rows),'primary_and_common_scores_checked':2*len(rows),
        'paired_metric_intervals_recomputed':intervals,'equal_pose_aggregates_checked':15,
        'comparison_sha256':hashlib.sha256(a.comparison.read_bytes()).hexdigest()}
    (out/'validation.json').write_text(json.dumps(validation,indent=2));print(json.dumps(validation,indent=2))

if __name__=='__main__':main()
