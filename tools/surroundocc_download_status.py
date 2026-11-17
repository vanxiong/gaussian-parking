"""Read-only progress for the validation sensor archive transfers."""
import json,time
from pathlib import Path
from install_surroundocc_inputs import REPORTS,DOWNLOADS

def main():
    entries=json.loads((REPORTS/'aws-archives.json').read_text())
    entries=[e for e in entries if 'trainval' in e['key'] and e['key'].endswith(('_keyframes.tgz','_blobs_lidar.tgz'))]
    total=sum(e['bytes'] for e in entries);transferred=0;completed=0;files=set()
    for entry in entries:
        name=Path(entry['key']).name;marker=REPORTS/(name+'.installed.json')
        if marker.exists():
            receipt=json.loads(marker.read_text());transferred+=entry['bytes'];completed+=1;files.update(receipt['files']);continue
        for path in [DOWNLOADS/name,DOWNLOADS/(name+'.partial')]:
            if path.exists():
                with path.open('rb') as stream:stream.seek(0,2);transferred+=stream.tell()
                break
    report={'completed_archives':completed,'total_archives':len(entries),'installed_sensor_files':len(files),
        'downloaded_archive_GiB':round(transferred/1024**3,2),'total_archive_GiB':round(total/1024**3,2),
        'download_percent':round(100*transferred/total,1)}
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
