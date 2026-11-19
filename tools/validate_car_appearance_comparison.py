"""Independently recompute repeated comparison metrics from saved raw predictions."""
import argparse,csv,hashlib,json,math,os
from pathlib import Path
import numpy as np

def local(value):
    if os.name=='nt' and value.startswith('/mnt/'):
        value=value[5].upper()+':/'+value[7:]
    return Path(value)

def main():
    p=argparse.ArgumentParser();p.add_argument('comparison',type=Path);a=p.parse_args()
    result=json.loads(a.comparison.read_text());out=a.comparison.parent
    n=result['request']['repeats_per_variant'];rows=result['runs']
    variants=tuple(result['request'].get('variants',['simple','realistic']))
    if len(rows)!=len(variants)*n:raise AssertionError('Wrong replay count')
    sensor=result['request'].get('experiment')=='sensor_inputs'
    truths={}
    for variant in variants:
        with np.load(local(result['sources'][variant]['ground_truth_npz'])) as gt:
            truths[variant]={k:gt[k].copy() for k in gt.files}
    mask=truths[variants[0]]['obstacles_above_ground']
    building=result['request'].get('experiment')=='building_presence'
    common=None
    if building:
        removed=truths['present']['occupied']&~truths['absent']['occupied'];common=mask&~removed
        if not np.array_equal(truths['present']['occupied'][common],truths['absent']['occupied'][common]):raise AssertionError('Common truth differs')
        with np.load(out/'comparison-masks.npz') as masks:
            if not np.array_equal(masks['common_region'],common) or not np.array_equal(masks['removed_occupied'],removed):raise AssertionError('Incorrect saved comparison mask')
    for r in rows:
        prediction=local(r['prediction'])
        if hashlib.sha256(prediction.read_bytes()).hexdigest()!=r['prediction_sha256']:raise AssertionError('Prediction checksum mismatch')
        raw=np.load(prediction)
        for scope,current_mask in [('primary',mask)]+([('common_region',common)] if building else []):
            truth=truths[r['variant']]['occupied'][current_mask];occupied=raw[current_mask]!=17
            tp=int(np.count_nonzero(occupied&truth));fp=int(np.count_nonzero(occupied&~truth))
            fn=int(np.count_nonzero(~occupied&truth));tn=int(np.count_nonzero(~occupied&~truth))
            expected={'tp':tp,'fp':fp,'fn':fn,'tn':tn,'occupied_iou':tp/(tp+fp+fn),'precision':tp/(tp+fp),'recall':tp/(tp+fn)}
            recorded=r if scope=='primary' else r[scope]
            for key,value in expected.items():
                if not math.isclose(value,recorded[key],rel_tol=0,abs_tol=1e-12):raise AssertionError('Replay metric mismatch: '+key)
            if tp+fp+fn+tn!=int(current_mask.sum()):raise AssertionError('Population changed')
    for variant in variants:
        subset=[r for r in rows if r['variant']==variant]
        if len(subset)!=n or {r['seed'] for r in subset}!=set(range(result['request']['first_seed'],result['request']['first_seed']+n)):raise AssertionError('Missing or duplicated seed')
        for scope in ['primary']+(['common_region'] if building else []):
            for key in ('occupied_iou','precision','recall'):
                values=[r[key] if scope=='primary' else r[scope][key] for r in subset];mean=math.fsum(values)/n
                sd=math.sqrt(math.fsum((v-mean)**2 for v in values)/(n-1))
                summary=result['summary' if scope=='primary' else 'common_region_summary'][variant][key]
                for field,value in [('mean',mean),('sample_sd',sd),('min',min(values)),('max',max(values))]:
                    if not math.isclose(summary[field],value,rel_tol=0,abs_tol=1e-12):raise AssertionError('Summary mismatch')
    for repeat in range(n):
        pair=[r for r in rows if r['repeat']==repeat]
        if len(pair)!=len(variants) or len({r['seed'] for r in pair})!=1 or {r['order_in_pair'] for r in pair}!=set(range(len(variants))):raise AssertionError('Pairing mismatch')
        order=[r['variant'] for r in sorted(pair,key=lambda r:r['order_in_pair'])]
        expected_order=result['execution_orders'][repeat%len(result['execution_orders'])] if sensor else list(variants if repeat%2==0 else variants[::-1])
        if order!=expected_order:raise AssertionError('Execution order mismatch')
    for scope in ['primary']+(['common_region'] if building else []):
        summaries=result['summary' if scope=='primary' else 'common_region_summary']
        comparisons=result['comparisons'] if sensor else {variants[1]:result['paired_differences' if scope=='primary' else 'common_region_paired_differences']}
        for variant,comparison in comparisons.items():
            for key,d in comparison.items():
                expected=summaries[variant][key]['mean']-summaries[variants[0]][key]['mean']
                if not math.isclose(d['mean'],expected,rel_tol=0,abs_tol=1e-12):raise AssertionError('Mean difference mismatch')
    with (out/'runs.csv').open(newline='') as stream:csv_rows=list(csv.DictReader(stream))
    if len(csv_rows)!=len(rows):raise AssertionError('CSV row count differs')
    for csv_row,row in zip(csv_rows,rows):
        for key in ['occupied_iou','precision','recall','tp','fp','fn','tn']:
            if float(csv_row[key])!=row[key]:raise AssertionError('CSV disagrees with JSON')
            if building and float(csv_row['common_'+key])!=row['common_region'][key]:raise AssertionError('Common-region CSV differs')
    validation={'passed':True,'recomputed_predictions':len(rows),'repeats_per_variant':n,
        'checked':'All prediction hashes, confusion counts, primary metrics, means, sample SDs, ranges, pairing/order, mean differences and CSV/JSON agreement',
        'evaluated_voxels_per_inference':int(mask.sum())}
    if building:validation['common_region_voxels_per_inference']=int(common.sum())
    (out/'validation.json').write_text(json.dumps(validation,indent=2));print(json.dumps(validation,indent=2))

if __name__=='__main__':main()
