"""Fetch only the authors' public validation metadata; reject HTML/login responses."""
import hashlib,http.cookiejar,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
URL='https://gtvault-my.sharepoint.com/:u:/g/personal/lgan31_gatech_edu/IQDMK0Iz4Z9HQKYcMsdkCivfAYp2MlcEtBTXFA921dANiJU?download=1'
def main():
    target=ROOT/'data/nuscenes_cam/nuscenes_infos_gf3d_val.pkl';target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise SystemExit('Metadata already exists; refusing overwrite')
    temporary=target.with_suffix('.partial');report={'source':URL,'downloaded':False}
    opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    try:
        with opener.open(urllib.request.Request(URL,headers={'User-Agent':'Mozilla/5.0'}),timeout=20) as response:
            first=response.read(8192)
            if 'html' in response.headers.get('Content-Type','').lower() or not first.startswith(b'\x80'):raise RuntimeError('SharePoint returned a web/login page instead of a validation pickle')
            size=len(first);digest=hashlib.sha256(first)
            with temporary.open('wb') as stream:
                stream.write(first)
                while chunk:=response.read(1024*1024):
                    size+=len(chunk)
                    if size>512*1024**2:raise RuntimeError('Unexpected metadata size exceeds 512 MiB')
                    stream.write(chunk);digest.update(chunk)
        temporary.rename(target);report.update(downloaded=True,bytes=size,sha256=digest.hexdigest())
    except Exception as exc:temporary.unlink(missing_ok=True);report['error']=str(exc)
    directory=ROOT/'reports/surroundocc';directory.mkdir(parents=True,exist_ok=True)
    (directory/'metadata-download.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not report['downloaded']:raise SystemExit(2)
if __name__=='__main__':main()
