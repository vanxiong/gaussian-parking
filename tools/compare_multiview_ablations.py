"""Unified repeated input/scene ablations across several fixed viewpoints."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import torch,cv2
from PIL import Image
from compare_car_appearance import digest,summarize,paired_summary,METRICS
from run_gaussianformer3d import ROOT,build_model,build_queries,infer_arrays
from prepare_capture import prepare,prepare_arrays
from occupancy_evaluation import SHAPE,metrics,confusion,voxelize_scene,evaluation_masks

VARIANTS=('simple','reference','no_building','high_resolution','current_sweep')
CONTRASTS={'appearance':('simple','reference'),'building_whole':('reference','no_building'),
           'building_common':('reference','no_building'),'resolution':('reference','high_resolution'),'history':('reference','current_sweep')}
ORDERS=[]
for shift in range(len(VARIANTS)):
    order=VARIANTS[shift:]+VARIANTS[:shift];ORDERS.extend([order,order[::-1]])

def close(a,b,tolerance=1e-6):return bool(np.allclose(a,b,atol=tolerance,rtol=0))

def load_inputs(request,out):
    arrays={};truths={};sources={};controls={};common_masks={};cache={}
    (out/'ground-truth').mkdir(exist_ok=True)
    for pose in request['poses']:
        pid=pose['id'];arrays[pid]={};truths[pid]={};sources[pid]={};checks={};metas={};geometries={}
        for variant,sample in request['samples'][pid].items():
            capture=Path(sample['capture']);meta=json.loads((capture/'capture.json').read_text());geo=json.loads(Path(sample['geometry']).read_text())
            if meta['ground_truth_snapshot']['used_as_model_input'] is not False:raise ValueError('Missing independent geometry provenance')
            if abs(meta['ground_truth_snapshot']['timestamp']-meta['simulation_timestamp_seconds'])>1e-6:raise ValueError('Snapshot timestamp mismatch')
            ego=np.asarray(meta['world_from_ego']);lidar=ego@np.asarray(meta['lidar']['ego_from_lidar'])
            if not close(ego,geo['world_from_ego'],1e-5):raise ValueError('Capture and exported ego poses differ')
            actual_yaw=math.degrees(math.atan2(ego[1,0],ego[0,0]));yaw_error=(actual_yaw-pose['yaw_deg']+180)%360-180
            if np.linalg.norm(ego[:2,3]-[pose['x'],pose['y']])>.05 or abs(yaw_error)>.5:raise ValueError('Wrong requested viewpoint')
            appearance=meta['appearance_ablation']
            if appearance['variant']!=('simple' if variant=='simple' else 'realistic') or appearance['selected_car_ids']!=list(range(10)):raise ValueError('Incorrect car-appearance variant')
            if variant!='simple' and len(appearance['cars'])!=10:raise ValueError('Missing detailed cars')
            building=any(o['path'].startswith('/World/Building/') for o in geo['objects'])
            if building!=(variant!='no_building'):raise ValueError('Wrong building presence')
            expected=[960,540] if variant=='high_resolution' else [640,360]
            if any(c['resolution']!=expected for c in meta['cameras']):raise ValueError('Wrong native resolution')
            if meta['lidar']['sweeps']!=11:raise ValueError('Expected eleven captured sweeps')
            key=json.dumps([geo['objects'],lidar.tolist()],sort_keys=True)
            if key not in cache:cache[key]=voxelize_scene(geo,lidar)
            truth=cache[key];masks=evaluation_masks(meta,geo)
            gt_path=out/'ground-truth'/f'{pid}-{variant}.npz'
            np.savez_compressed(gt_path,occupied=truth,**masks,world_from_lidar=lidar)
            truths[pid][variant]={'occupied':truth,**masks,'world_from_lidar':lidar}
            arrays[pid][variant]=prepare(capture);metas[variant]=meta;geometries[variant]=geo
            sources[pid][variant]={'capture':str(capture),'geometry':sample['geometry'],'ground_truth_npz':str(gt_path),
                'capture_sha256':digest(capture/'capture.json'),'scene_sha256':geo['scene_sha256'],
                'lidar_sha256':digest(capture/meta['lidar']['points']),
                'rgb_sha256':{c['name']:digest(capture/c['image']) for c in meta['cameras']}}
        reference=metas['reference'];base_gt=truths[pid]['reference']
        for variant in ['simple','no_building','high_resolution']:
            checks[variant+'_ego_pose_matches']=close(metas[variant]['world_from_ego'],reference['world_from_ego'])
            checks[variant+'_lidar_extrinsics_match']=close(metas[variant]['lidar']['ego_from_lidar'],reference['lidar']['ego_from_lidar'])
            cams={c['name']:c for c in metas[variant]['cameras']};valid=True
            for c in reference['cameras']:
                other=cams[c['name']];k=np.asarray(c['intrinsic']).copy()
                if variant=='high_resolution':k[:2]*=1.5
                valid&=close(c['ego_from_optical'],other['ego_from_optical']) and close(k,other['intrinsic'])
            checks[variant+'_camera_calibration_matches']=bool(valid)
            checks[variant+'_primary_mask_identical']=np.array_equal(base_gt['obstacles_above_ground'],truths[pid][variant]['obstacles_above_ground'])
        for variant in ['simple','high_resolution']:
            checks[variant+'_points_identical']=np.array_equal(arrays[pid][variant]['points'],arrays[pid]['reference']['points'])
            checks[variant+'_truth_identical']=np.array_equal(base_gt['occupied'],truths[pid][variant]['occupied'])
            checks[variant+'_reference_objects_identical']=geometries[variant]['objects']==geometries['reference']['objects']
        for variant in ['no_building','high_resolution']:
            checks[variant+'_realistic_cars_identical']=metas[variant]['appearance_ablation']==reference['appearance_ablation']
        def nonbuilding(v):return {o['path']:o for o in geometries[v]['objects'] if not o['path'].startswith('/World/Building/')}
        checks['nonbuilding_objects_identical']=nonbuilding('reference')==nonbuilding('no_building')
        absent=truths[pid]['no_building']['occupied'];removed=base_gt['occupied']&~absent
        checks['removal_adds_no_occupancy']=not np.any(absent&~base_gt['occupied'])
        common=base_gt['obstacles_above_ground']&~removed;common_masks[pid]=common
        checks['common_truth_identical']=np.array_equal(base_gt['occupied'][common],absent[common])
        if not all(checks.values()):raise ValueError(pid+' controls failed: '+json.dumps(checks))
        capture=Path(sources[pid]['reference']['capture']);points=arrays[pid]['reference']['points'];current=points[np.isclose(points[:,4],0)].copy()
        if len(current)!=reference['lidar']['current_point_count']:raise ValueError('Current sweep selection mismatch')
        rgb={c['name']:np.asarray(Image.open(capture/c['image']).convert('RGB')) for c in reference['cameras']}
        arrays[pid]['current_sweep']=prepare_arrays(reference,rgb,current)
        for name in ['img','projection_mat','image_wh','dpt']:
            checks['current_'+name+'_identical']=np.array_equal(arrays[pid]['current_sweep'][name],arrays[pid]['reference'][name])
            if not checks['current_'+name+'_identical']:raise ValueError('Current sweep changed other inputs')
        sources[pid]['current_sweep']={**sources[pid]['reference'],'point_input_transform':'Retain exactly zero time-lag points'}
        truths[pid]['current_sweep']=base_gt
        mask_path=out/'ground-truth'/f'{pid}-common-mask.npz';np.savez_compressed(mask_path,common_region=common,removed_occupied=removed)
        controls[pid]={'checks':checks,'common_mask':str(mask_path),'requested_pose':pose,'world_from_ego':reference['world_from_ego'],
            'primary_voxels':int(base_gt['obstacles_above_ground'].sum()),'common_voxels':int(common.sum()),
            'occupied_voxels':{v:int((truths[pid][v]['occupied']&base_gt['obstacles_above_ground']).sum()) for v in VARIANTS},
            'input_point_counts':{v:len(arrays[pid][v]['points']) for v in VARIANTS}}
        print('POSE_CONTROLS_PASSED',pid,json.dumps(controls[pid]),flush=True)
    return arrays,truths,sources,controls,common_masks

def summarize_results(result):
    rows=result['runs'];per_pose={}
    for pose in result['request']['poses']:
        pid=pose['id'];ordered={v:sorted([r for r in rows if r['pose']==pid and r['variant']==v],key=lambda r:r['seed']) for v in VARIANTS}
        conditions={v:{k:summarize([r[k] for r in ordered[v]]) for k in METRICS} for v in VARIANTS}
        effects={}
        for contrast,(first,second) in CONTRASTS.items():
            scope='common_region' if contrast=='building_common' else None
            effects[contrast]={k:paired_summary([r[scope][k] if scope else r[k] for r in ordered[first]],[r[scope][k] if scope else r[k] for r in ordered[second]]) for k in METRICS}
        per_pose[pid]={'conditions':conditions,'effects':effects}
    result['per_pose']=per_pose;across={}
    for contrast in CONTRASTS:
        across[contrast]={}
        for k in METRICS:
            values={pid:d['effects'][contrast][k]['mean'] for pid,d in per_pose.items()}
            across[contrast][k]={'equal_pose_mean':float(np.mean(list(values.values()))),
                'between_pose_sample_sd':float(np.std(list(values.values()),ddof=1)),'min_pose_effect':float(min(values.values())),
                'max_pose_effect':float(max(values.values())),
                'positive_poses':[pid for pid,v in values.items() if v>0],'negative_poses':[pid for pid,v in values.items() if v<0],
                'positive_interval_poses':[pid for pid,d in per_pose.items() if d['effects'][contrast][k]['mean_95pct_t_interval'][0]>0],
                'negative_interval_poses':[pid for pid,d in per_pose.items() if d['effects'][contrast][k]['mean_95pct_t_interval'][1]<0],
                'scope':'Four selected viewpoints in one fixed lot; no across-scene inferential confidence interval'}
    result['across_pose']=across

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);a=p.parse_args();request=json.loads(a.manifest.read_text());out=a.manifest.parent
    arrays,truths,sources,controls,common=load_inputs(request,out);(out/'controls.json').write_text(json.dumps(controls,indent=2))
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.manual_seed(request['first_seed']);np.random.seed(request['first_seed']);model,config,tensors=build_model(anchor_assignment=request['anchor_assignment']);queries=build_queries(config)
    if tensors!=941 or any(p.requires_grad for p in model.parameters()):raise RuntimeError('Expected strict frozen checkpoint')
    for pose in request['poses']:
        for variant in VARIANTS:
            for _ in range(request['warmup_per_condition']):infer_arrays(model,config,arrays[pose['id']][variant],queries)
    (out/'predictions').mkdir(exist_ok=True);rows=[];pose_ids=[p['id'] for p in request['poses']]
    result={'request':request,'sources':sources,'controls':controls,'variant_orders':[list(o) for o in ORDERS],
        'strict_tensor_count':tensors,'checkpoint_sha256':digest(ROOT/'checkpoints/surroundocc_release.pth'),'device':torch.cuda.get_device_name(),
        'ground_truth_used_as_model_input':False,'runs':rows}
    for repeat in range(request['repeats_per_condition']):
        seed=request['first_seed']+repeat;shift=repeat%len(pose_ids);pose_order=pose_ids[shift:]+pose_ids[:shift]
        if repeat%2:pose_order=pose_order[::-1]
        for pid in pose_order:
            for position,variant in enumerate(ORDERS[repeat%len(ORDERS)]):
                torch.manual_seed(seed);np.random.seed(seed);raw,_,timing=infer_arrays(model,config,arrays[pid][variant],queries)
                if raw.shape!=SHAPE or raw.max()>17:raise ValueError('Invalid prediction')
                path=out/'predictions'/f'{pid}-{variant}-seed-{seed}.npy';np.save(path,raw);gt=truths[pid][variant]
                score=metrics(confusion(raw!=17,gt['occupied'],gt['obstacles_above_ground']))
                secondary=metrics(confusion(raw!=17,gt['occupied'],common[pid]))
                if any(score[k] is None or secondary[k] is None for k in METRICS):raise ValueError('Undefined primary/common metric')
                rows.append({'pose':pid,'variant':variant,'repeat':repeat,'seed':seed,'order_in_condition_group':position,
                    'prediction':str(path),'prediction_sha256':digest(path),**score,'common_region':secondary,**timing})
                if len(rows)%5==0:print('REPLAY_GROUP',repeat+1,pid,'completed',len(rows),flush=True)
            (out/'partial-runs.json').write_text(json.dumps(rows,indent=2))
    summarize_results(result);(out/'comparison.json').write_text(json.dumps(result,indent=2))
    keys=['pose','variant','repeat','seed','order_in_condition_group','tp','fp','fn','tn',*METRICS,*['common_'+k for k in ['tp','fp','fn','tn',*METRICS]],'prediction','prediction_sha256','inference_seconds']
    with (out/'runs.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore');writer.writeheader()
        for r in rows:writer.writerow({**r,**{'common_'+k:v for k,v in r['common_region'].items()}})
    from multiview_ablation_report import build_report
    build_report(out,result)
    print('VIEWPOINT_RESULTS_READY',out/'VIEWPOINT_RESULTS.md',flush=True)

if __name__=='__main__':main()
