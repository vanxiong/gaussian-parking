"""Reproducible public-source acquisition; no credentials and no training data retained."""
import argparse,concurrent.futures,json,subprocess,sys,urllib.request
from pathlib import Path
from install_surroundocc_inputs import ROOT,DATA,DOWNLOADS,REPORTS,metadata,infos,depth,labels
from check_surroundocc_data import inspect_data

AWS='https://motional-nuscenes.s3.ap-northeast-1.amazonaws.com/'
INFO='https://cloud.tsinghua.edu.cn/d/bb96379a3e46442c8898/files/?p=%2Fnuscenes_infos_val_sweeps_occ.pkl&dl=1'
DEPTH='https://drive.usercontent.google.com/download?id=1C0lYFU1Wu2fVOjC-YcUBThS5rf4ehieR&export=download&confirm=t'

def transfer(url,name,size):
    target=DOWNLOADS/name
    if target.exists():
        if target.stat().st_size!=size:raise RuntimeError('Existing archive has unexpected size: '+str(target))
        return
    partial=target.with_suffix(target.suffix+'.partial')
    with (REPORTS/(name+'.download.log')).open('w') as log:
        subprocess.run(['curl.exe','-L','--fail','--retry','5','--connect-timeout','30','-C','-',
                        '-o',str(partial),url],stdout=log,stderr=subprocess.STDOUT,check=True)
    if partial.stat().st_size!=size:raise RuntimeError('Wrong download size; refusing to install: '+name)
    partial.replace(target)

def ready(name,paths):return (REPORTS/(name+'-installed.json')).exists() and all(p.is_file() for p in paths)

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    if not 1<=args.workers<=6:p.error('--workers must be 1..6')
    DOWNLOADS.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True)
    subprocess.run([str(ROOT/'evaluation/setup_surroundocc.bat')],check=True)
    subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-deps',
        '--cache-dir',str(ROOT/'.cache/download-pip'),'--target',str(ROOT/'.download-deps'),
        '-r',str(ROOT/'requirements-download-lock.txt')],check=True)
    transfer(AWS+'public/v1.0/v1.0-trainval_meta.tgz','v1.0-trainval_meta.tgz',461678030)
    if not ready('metadata',[DATA/'nuscenes/v1.0-trainval'/f'{name}.json' for name in ['scene','sample','sample_data','ego_pose','calibrated_sensor','sensor']]):metadata()
    transfer(INFO,'nuscenes_infos_val_sweeps_occ.pkl',121003437)
    if not (DATA/'nuscenes_cam/nuscenes_infos_gf3d_val.pkl').is_file():infos()
    selection=json.loads((DATA/'validation-files.json').read_text())
    def depth_stage():
        if ready('depth',[DATA/'nuscenes/depth_gt'/n for n in selection['depths']]):return
        transfer(DEPTH,'depth_gt.zip',7744905988);depth()
    def label_stage():
        if ready('labels',[DATA/'surroundocc/samples'/n for n in selection['labels']]):return
        url='https://huggingface.co/api/datasets/sangramrout/surroundOCC/tree/main?recursive=false&limit=100'
        with urllib.request.urlopen(url,timeout=30) as response:entries=json.load(response)
        entries=[e for e in entries if e['path'].endswith('.tar.zst')]
        (REPORTS/'label-mirror-files.json').write_text(json.dumps(entries,indent=2))
        for entry in entries:transfer('https://huggingface.co/datasets/sangramrout/surroundOCC/resolve/main/'+entry['path']+'?download=true',entry['path'],entry['size'])
        labels()
    def sensor_stage():
        if not (REPORTS/'aws-archives.json').is_file():
            import xml.etree.ElementTree as ET
            with urllib.request.urlopen(AWS+'?list-type=2&prefix=public/v1.0/&delimiter=/',timeout=30) as response:tree=ET.fromstring(response.read())
            ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
            entries=[{'key':e.find('s:Key',ns).text,'bytes':int(e.find('s:Size',ns).text)} for e in tree.findall('s:Contents',ns)]
            (REPORTS/'aws-archives.json').write_text(json.dumps(entries,indent=2))
        subprocess.run([sys.executable,str(ROOT/'tools/download_surroundocc_sensors.py'),'--workers',str(args.workers)],check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(job) for job in [depth_stage,label_stage,sensor_stage]]
        for future in concurrent.futures.as_completed(futures):future.result()
    report=inspect_data();(REPORTS/'data-preflight.json').write_text(json.dumps(report,indent=2))
    if not report['ready']:raise RuntimeError('Data preflight failed after downloads; inspect its report')
    print('SURROUNDOCC_VALIDATION_DATA_READY',flush=True)
if __name__=='__main__':main()
