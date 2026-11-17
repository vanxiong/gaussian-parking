"""Download the authors' released nuScenes checkpoint without logging cookies."""
import argparse
import hashlib
import http.cookiejar
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://gtvault-my.sharepoint.com/:u:/g/personal/lgan31_gatech_edu/IQCoIH-VSc5tRrAxVbg8Pm4PARtX4wL0DGgD5qIbWYHFFRI?download=1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=URL)
    args = parser.parse_args()
    target = ROOT / 'checkpoints/surroundocc_release.pth'
    if target.exists():
        raise SystemExit('Checkpoint already exists; refusing to overwrite it.')
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix('.partial')
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    request = urllib.request.Request(args.url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with opener.open(request, timeout=45) as response:
            content_type = response.headers.get('Content-Type', '')
            first = response.read(8192)
            if 'text/html' in content_type or not (first.startswith(b'PK\x03\x04') or first.startswith(b'\x80')):
                raise RuntimeError('Server returned a web page or unsupported content, not a PyTorch checkpoint.')
            digest = hashlib.sha256(first)
            size = len(first)
            with temporary.open('wb') as output:
                output.write(first)
                while chunk := response.read(4 * 1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
        if size < 1_000_000:
            raise RuntimeError('Checkpoint is unexpectedly small; download rejected.')
        temporary.rename(target)
        report = {'file': target.name, 'bytes': size, 'sha256': digest.hexdigest(),
                  'source': args.url, 'tensor_loading_verified': False}
        (ROOT / 'reports/checkpoint.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == '__main__':
    main()
