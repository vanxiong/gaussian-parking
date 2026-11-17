"""Fail-closed preflight: enumerate every required validation input before using CUDA."""
import argparse,hashlib,json,pickle,shutil
from pathlib import Path
import numpy as np
from surroundocc_protocol import ROOT,DATA,EXPECTED_SAMPLES,confined,resolve_sensor

def inspect_data(base=DATA):
    base=confined(base);root=base/'nuscenes';info=base/'nuscenes_cam/nuscenes_infos_gf3d_val.pkl';occ=base/'surroundocc/samples'
    report={'ready':False,'data_root':str(base),'free_bytes':shutil.disk_usage(ROOT).free,
        'expected_validation_samples':EXPECTED_SAMPLES,'missing':[],'required_files':{},'measured_benchmark':False}
    scene_file=root/'v1.0-trainval/scene.json';sample_file=root/'v1.0-trainval/sample.json'
    required={'metadata':info,'camera_samples':root/'samples','lidar_sweeps':root/'sweeps/LIDAR_TOP','depth_inputs':root/'depth_gt','occupancy_labels':occ,
              'nuscenes_scene_metadata':scene_file,'nuscenes_sample_metadata':sample_file}
    for name,path in required.items():
        if not path.exists():report['missing'].append({'kind':name,'path':str(path)})
    if not info.is_file():return report
    report['metadata_sha256']=hashlib.sha256(info.read_bytes()).hexdigest()
    with info.open('rb') as stream:data=pickle.load(stream)
    if not isinstance(data,dict) or not isinstance(data.get('infos'),dict) or not isinstance(data.get('metadata'),(list,tuple,np.ndarray)):raise ValueError('Expected the GaussianFormer3D authors infos/metadata schema, not another nuScenes pickle')
    keys=sorted(data['metadata'],key=lambda x:x[0]+f'{x[1]:0>3}');report['metadata_samples']=len(keys)
    if len(keys)!=EXPECTED_SAMPLES:report['missing'].append({'kind':'validation_split','error':f'Expected {EXPECTED_SAMPLES} keyframes, found {len(keys)}'})
    paths={};tokens=[];scenes=set()
    def add(path,kind):paths[str(path)]=kind
    for scene,index in keys:
        sample=data['infos'][scene][index];tokens.append(sample['token']);scenes.add(scene)
        for camera in ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
            image=resolve_sensor(sample['data'][camera]['filename'],root);add(image,'camera')
            add(root/'depth_gt'/(image.name+'.bin'),'depth')
        lidar=resolve_sensor(sample.get('lidar_path',sample['data']['LIDAR_TOP']['filename']),root);add(lidar,'lidar')
        add(occ/(Path(sample['data']['LIDAR_TOP']['filename']).name+'.npy'),'occupancy')
        if 'lidar_path' not in sample or 'sweeps' not in sample:raise ValueError('Author validation metadata must include lidar_path and sweeps')
        for sweep in sample['sweeps']:add(resolve_sensor(sweep['data_path'],root),'sweep')
    if len(set(tokens))!=len(tokens):raise ValueError('Duplicate validation sample tokens')
    official=json.loads((ROOT/'configs/surroundocc_val_scenes.json').read_text())
    report['official_split_source_sha256']=official['source_sha256']
    if scene_file.is_file() and sample_file.is_file():
        records=json.loads(scene_file.read_text());expected_names=set(official['scene_names'])
        scene_tokens={row['token'] for row in records if row['name'] in expected_names}
        expected_tokens={row['token'] for row in json.loads(sample_file.read_text()) if row['scene_token'] in scene_tokens}
        valid=scenes==scene_tokens and len(scene_tokens)==150 and set(tokens)==expected_tokens and len(expected_tokens)==EXPECTED_SAMPLES
        report['official_split_verified']=valid
        if not valid:report['missing'].append({'kind':'split_membership','error':'Sample tokens or scenes do not exactly match the official validation split'})
    else:report['official_split_verified']=False
    report.update(unique_scenes=len(scenes),sample_token_sha256=hashlib.sha256('\n'.join(tokens).encode()).hexdigest())
    missing_counts={};missing_examples=[];present_bytes=0
    for value,kind in paths.items():
        report['required_files'][kind]=report['required_files'].get(kind,0)+1
        path=Path(value)
        if not path.is_file():
            missing_counts[kind]=missing_counts.get(kind,0)+1
            if len(missing_examples)<25:missing_examples.append(value)
        else:present_bytes+=path.stat().st_size
    if missing_counts:report['missing'].append({'kind':'sample_files','counts':missing_counts,'examples':missing_examples})
    report.update(present_input_bytes=present_bytes,ready=not report['missing'])
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=DATA);args=p.parse_args()
    output=ROOT/'reports/surroundocc';output.mkdir(parents=True,exist_ok=True)
    try:report=inspect_data(args.data_root)
    except Exception as exc:report={'ready':False,'error':str(exc),'measured_benchmark':False}
    (output/'data-preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not report['ready']:raise SystemExit(2)
if __name__=='__main__':main()
