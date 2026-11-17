"""Download official archives concurrently; retain only validation files."""
import argparse,concurrent.futures,hashlib,json,math,shutil,subprocess,tarfile,threading,time,urllib.request
from pathlib import Path
from install_surroundocc_inputs import DATA,ROOT,DOWNLOADS,REPORTS,copy_member,record

BASE='https://motional-nuscenes.s3.ap-northeast-1.amazonaws.com/'
LOCK=threading.Lock()
def install(entry,needed):
    name=Path(entry['key']).name;url=BASE+entry['key'];archive=DOWNLOADS/name;marker=REPORTS/(name+'.installed.json')
    if marker.exists():return json.loads(marker.read_text())
    request=urllib.request.Request(url,method='HEAD')
    with urllib.request.urlopen(request,timeout=30) as response:etag=response.headers['ETag'].strip('"')
    print('ARCHIVE_START',name,entry['bytes'],flush=True)
    if not archive.is_file() or archive.stat().st_size!=entry['bytes']:
        if archive.exists():raise RuntimeError('Unexpected incomplete final archive; only .partial files are resumed')
        temporary=archive.with_suffix(archive.suffix+'.partial')
        with (REPORTS/(name+'.transfer.log')).open('w') as log:
            subprocess.run(['curl.exe','-L','--fail','--retry','5','--retry-delay','2','--connect-timeout','30','-C','-',
                            '-o',str(temporary),url],stdout=log,stderr=subprocess.STDOUT,check=True)
        if temporary.stat().st_size!=entry['bytes']:raise RuntimeError('Archive size mismatch: '+name)
        temporary.replace(archive)
    md5=hashlib.md5();part_digests=[]
    with archive.open('rb') as stream:
        while chunk:=stream.read(8*1024**2):md5.update(chunk);part_digests.append(hashlib.md5(chunk).digest())
    actual=hashlib.md5(b''.join(part_digests)).hexdigest()+'-'+str(len(part_digests)) if '-' in etag else md5.hexdigest()
    if actual!=etag:raise RuntimeError('Official S3 object checksum mismatch: '+name)
    found=[];written=0
    with tarfile.open(archive,'r:gz') as source:
        for member in source:
            if not member.isfile():continue
            path=Path(member.name)
            parts=path.parts
            # Original archives may have an optional leading ./ or nuscenes/.
            if 'samples' in parts:position=parts.index('samples')
            elif 'sweeps' in parts:position=parts.index('sweeps')
            else:continue
            suffix=Path(*parts[position:]).as_posix()
            if suffix not in needed:continue
            with source.extractfile(member) as stream,LOCK:written+=copy_member(stream,DATA/'nuscenes'/suffix,member.size)
            found.append(suffix)
    result={'archive':name,'url':url,'compressed_bytes':entry['bytes'],'s3_etag':etag,
            'checksum_verified':True,'selected_files':len(found),'installed_files':written,'files':found}
    marker.write_text(json.dumps(result,indent=2))
    # These are tool-created transfer caches, validated above. Keep original data,
    # logs and checksum receipts, and release each multi-GB archive after extraction.
    archive.resolve().relative_to(DOWNLOADS.resolve());archive.unlink()
    print('ARCHIVE_COMPLETE',name,len(found),flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    if not 1<=args.workers<=6:p.error('workers must be 1..6')
    needed=set(json.loads((DATA/'validation-files.json').read_text())['sensors'])
    entries=json.loads((REPORTS/'aws-archives.json').read_text())
    entries=[e for e in entries if 'trainval' in e['key'] and (e['key'].endswith('_keyframes.tgz') or e['key'].endswith('_blobs_lidar.tgz'))]
    entries.sort(key=lambda e:(not e['key'].endswith('_keyframes.tgz'),e['key']))
    if shutil.disk_usage(ROOT).free<80*1024**3:raise RuntimeError('Need at least 80 GiB free for concurrent archive caches and selected sensor files')
    started=time.time();done=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(install,e,needed):e for e in entries}
        for future in concurrent.futures.as_completed(futures):
            result=future.result();done.append(result)
            record('sensor-download-progress',{'completed_archives':len(done),'total_archives':len(entries),
                'elapsed_seconds':time.time()-started,'selected_files':sum(r['selected_files'] for r in done)})
    missing=[name for name in sorted(needed) if not (DATA/'nuscenes'/name).is_file()]
    record('sensors-installed',{'source':'Official Motional nuScenes S3 distribution','required_files':len(needed),
        'missing':missing,'downloaded_compressed_bytes':sum(e['bytes'] for e in entries),'elapsed_seconds':time.time()-started,
        'archives':[{'archive':r['archive'],'s3_etag':r['s3_etag'],'selected_files':r['selected_files']} for r in done]})
    if missing:raise RuntimeError(f'{len(missing)} sensor files still missing')
    print('ALL_VALIDATION_SENSORS_READY',len(needed),flush=True)
if __name__=='__main__':main()
