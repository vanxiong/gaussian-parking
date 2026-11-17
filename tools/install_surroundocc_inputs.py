"""Validate archives, install only validation data, and record every source."""
import argparse,hashlib,json,pickle,shutil,sys,tarfile,zipfile
from pathlib import Path
import numpy as np
from surroundocc_protocol import ROOT,DATA,EXPECTED_SAMPLES,resolve_sensor,confined

DOWNLOADS=DATA/'downloads'
REPORTS=ROOT/'reports/surroundocc'
def record(name,report):
    REPORTS.mkdir(parents=True,exist_ok=True);(REPORTS/(name+'.json')).write_text(json.dumps(report,indent=2))

def copy_member(stream,target,size):
    target=confined(target)
    if target.is_file():
        if target.stat().st_size!=size:raise ValueError('Existing file size mismatch: '+str(target))
        return False
    if shutil.disk_usage(ROOT).free<size+8*1024**3:raise RuntimeError('Insufficient free disk space')
    target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_suffix(target.suffix+'.partial')
    with temporary.open('wb') as output:shutil.copyfileobj(stream,output,1024*1024)
    if temporary.stat().st_size!=size:raise ValueError('Truncated archive member')
    temporary.replace(target);return True

def metadata():
    archive=DOWNLOADS/'v1.0-trainval_meta.tgz'
    with archive.open('rb') as stream:digest=hashlib.file_digest(stream,'md5').hexdigest()
    parts=[]
    with archive.open('rb') as stream:
        while chunk:=stream.read(8*1024**2):parts.append(hashlib.md5(chunk).digest())
    etag=hashlib.md5(b''.join(parts)).hexdigest()+'-'+str(len(parts))
    # The current S3 object is a newer archive than the legacy md5.checksum entry.
    # Verify its multipart digest against the ETag returned by official S3 HEAD.
    if etag!='fd4ea76d8701fb567a67a65025d0b022-56':raise ValueError('Official S3 metadata object ETag mismatch')
    installed=[]
    with tarfile.open(archive,'r:gz') as source:
        for member in source:
            path=Path(member.name)
            if member.isfile() and path.parts[0]=='v1.0-trainval' and path.suffix=='.json':
                with source.extractfile(member) as stream:copy_member(stream,DATA/'nuscenes'/path,member.size)
                installed.append(member.name)
    record('metadata-installed',{'archive_md5':digest,'official_s3_multipart_etag':etag,'official_checksum_verified':True,
        'legacy_md5_note':'The md5.checksum entry differs from the current Jan 30 2024 S3 object; multipart ETag verified instead.','files':installed})
    print('INSTALLED_METADATA',len(installed),flush=True)

def infos():
    source=DOWNLOADS/'nuscenes_infos_val_sweeps_occ.pkl'
    with source.open('rb') as stream:data=pickle.load(stream)
    if not isinstance(data,dict) or not isinstance(data.get('infos'),dict):raise ValueError('Incorrect metadata schema')
    keys=data['metadata']
    if len(keys)!=EXPECTED_SAMPLES or len({s for s,i in keys})!=150:raise ValueError('Incorrect validation count')
    if any('sweeps' not in data['infos'][s][i] for s,i in keys):
        regenerate_sweeps(data)
    needed=set();depths=set();labels=set();missing_fields=set()
    for scene,index in keys:
        sample=data['infos'][scene][index]
        for camera in ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
            path=resolve_sensor(sample['data'][camera]['filename'],DATA/'nuscenes')
            needed.add(path.relative_to(DATA/'nuscenes').as_posix());depths.add(path.name+'.bin')
        lidar=resolve_sensor(sample['data']['LIDAR_TOP']['filename'],DATA/'nuscenes')
        sample['lidar_path']='data/nuscenes/'+lidar.relative_to(DATA/'nuscenes').as_posix()
        needed.add(lidar.relative_to(DATA/'nuscenes').as_posix());labels.add(lidar.name+'.npy')
        if 'sweeps' not in sample:missing_fields.add('sweeps')
        for sweep in sample.get('sweeps',[]):
            path=resolve_sensor(sweep['data_path'],DATA/'nuscenes');suffix=path.relative_to(DATA/'nuscenes').as_posix()
            sweep['data_path']='data/nuscenes/'+suffix;needed.add(suffix)
    target=DATA/'nuscenes_cam/nuscenes_infos_gf3d_val.pkl'
    if missing_fields:raise RuntimeError('Metadata needs official sensor-history regeneration: '+str(missing_fields))
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise RuntimeError('Refusing to overwrite existing validation metadata')
    with target.open('wb') as stream:pickle.dump(data,stream,protocol=4)
    selection={'sensors':sorted(needed),'depths':sorted(depths),'labels':sorted(labels)}
    (DATA/'validation-files.json').write_text(json.dumps(selection,indent=2))
    with source.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    record('infos-installed',{'source':'https://cloud.tsinghua.edu.cn/d/bb96379a3e46442c8898/files/?p=%2Fnuscenes_infos_val_sweeps_occ.pkl&dl=1',
        'source_sha256':digest,'samples':len(keys),'scenes':150,'conversion':'Camera/calibration/pose metadata retained; missing sweeps regenerated with GF3D preparation helper from official nuScenes tables; paths relocated',
        'selection_counts':{k:len(v) for k,v in selection.items()}})
    print('INSTALLED_INFOS',len(keys),{k:len(v) for k,v in selection.items()},flush=True)

def regenerate_sweeps(data):
    """Apply the existing GF3D preparation routine to original nuScenes tables."""
    import importlib.util
    sys.path.insert(0,str(ROOT/'.download-deps'));sys.path.insert(0,str(ROOT/'.eval-deps'))
    import ijson
    from pyquaternion import Quaternion
    folder=DATA/'nuscenes/v1.0-trainval'
    tokens={data['infos'][s][i]['token'] for s,i in data['metadata']}
    sensors={r['token']:r for r in json.loads((folder/'sensor.json').read_text())}
    calibrations={r['token']:r for r in json.loads((folder/'calibrated_sensor.json').read_text())}
    lidar_calibrations={token for token,row in calibrations.items() if sensors[row['sensor_token']]['channel']=='LIDAR_TOP'}
    with (folder/'sample_data.json').open('rb') as stream:
        records={row['token']:row for row in ijson.items(stream,'item',use_float=True) if row['sample_token'] in tokens and row['calibrated_sensor_token'] in lidar_calibrations}
    poses_needed={row['ego_pose_token'] for row in records.values()}
    with (folder/'ego_pose.json').open('rb') as stream:
        poses={row['token']:row for row in ijson.items(stream,'item',use_float=True) if row['token'] in poses_needed}
    print('INDEXED_METADATA',len(records),'LiDAR records',len(poses),'poses',flush=True)
    class Tables:
        def get(self,table,token):return {'sample_data':records,'calibrated_sensor':calibrations,'ego_pose':poses}[table][token]
        def get_sample_data_path(self,token):return str(resolve_sensor(records[token]['filename'],DATA/'nuscenes'))
    helper=ROOT/'GaussianFormer3D/tools/make_gf3d_infos.py'
    spec=importlib.util.spec_from_file_location('gf3d_info_preparation',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tables=Tables()
    for scene,index in data['metadata']:
        sample=data['infos'][scene][index];sd=records[sample['data']['LIDAR_TOP']['token']]
        calibration=calibrations[sd['calibrated_sensor_token']];pose=poses[sd['ego_pose_token']];sweeps=[]
        while len(sweeps)<10 and sd['prev']:
            sweeps.append(module.obtain_sensor2top(tables,sd['prev'],np.asarray(calibration['translation']),Quaternion(calibration['rotation']).rotation_matrix,
                np.asarray(pose['translation']),Quaternion(pose['rotation']).rotation_matrix,'lidar'))
            sd=records[sd['prev']]
        sample['sweeps']=sweeps
    record('sweep-regeneration',{'source':'Official nuScenes trainval metadata + existing GF3D obtain_sensor2top helper',
        'helper_sha256':hashlib.sha256(helper.read_bytes()).hexdigest(),'max_sweeps':10,'samples':len(tokens),
        'reason':'Working SurroundOcc metadata contains full camera/pose/calibration records but no GF3D LiDAR sweeps; generated using its supplied routine.'})

def depth():
    needed=set(json.loads((DATA/'validation-files.json').read_text())['depths']);found=set();installed=0
    archive=DOWNLOADS/'depth_gt.zip'
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            name=Path(member.filename).name
            if name not in needed:continue
            if member.file_size%12:raise ValueError('Invalid x/y/depth triple file size')
            with source.open(member) as stream:installed+=copy_member(stream,DATA/'nuscenes/depth_gt'/name,member.file_size)
            found.add(name)
    record('depth-installed',{'source':'https://drive.google.com/file/d/1C0lYFU1Wu2fVOjC-YcUBThS5rf4ehieR/view','selected_files':len(found),'missing':sorted(needed-found),'installed':installed})
    if found!=needed:raise RuntimeError('Depth archive does not cover all validation images')
    print('INSTALLED_DEPTH',len(found),flush=True)

def labels():
    sys.path.insert(0,str(ROOT/'.download-deps'));import zstandard
    needed=set(json.loads((DATA/'validation-files.json').read_text())['labels']);found=set()
    files=json.loads((REPORTS/'label-mirror-files.json').read_text());records=[]
    for entry in files:
        path=DOWNLOADS/entry['path']
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=entry['lfs']['oid'] or path.stat().st_size!=entry['size']:raise ValueError('Label archive checksum mismatch')
        count=0
        with path.open('rb') as raw,zstandard.ZstdDecompressor().stream_reader(raw) as decoded,tarfile.open(fileobj=decoded,mode='r|') as source:
            for member in source:
                name=Path(member.name).name
                if member.isfile() and name in needed:
                    with source.extractfile(member) as stream:copy_member(stream,DATA/'surroundocc/samples'/name,member.size)
                    found.add(name);count+=1
        records.append({'archive':entry['path'],'sha256':digest,'selected_files':count});print('INSTALLED_LABEL_SHARD',entry['path'],count,flush=True)
    record('labels-installed',{'mirror':'https://huggingface.co/datasets/sangramrout/surroundOCC','archives':records,'selected_files':len(found),'missing':sorted(needed-found)})
    if found!=needed:raise RuntimeError('Label mirror does not cover all validation keyframes')

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['metadata','infos','depth','labels']);args=p.parse_args()
    globals()[args.stage]()
if __name__=='__main__':main()
