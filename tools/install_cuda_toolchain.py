"""Install NVIDIA's CUDA compiler/header redistributables locally on Linux."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://developer.download.nvidia.com/compute/cuda/redist/'
manifest = json.load(urllib.request.urlopen(BASE + 'redistrib_12.8.1.json', timeout=45))
cache = ROOT / '.cache/cuda-redist'
prefix = ROOT / '.toolchains/cuda'
cache.mkdir(parents=True, exist_ok=True)
prefix.mkdir(parents=True, exist_ok=True)
for component in ['cuda_nvcc', 'cuda_cudart', 'cuda_cccl']:
    item = manifest[component]['linux-x86_64']
    archive = cache / Path(item['relative_path']).name
    if not archive.exists():
        with urllib.request.urlopen(BASE + item['relative_path'], timeout=90) as source, archive.open('wb') as target:
            shutil.copyfileobj(source, target)
    with archive.open('rb') as handle:
        if hashlib.file_digest(handle, 'sha256').hexdigest() != item['sha256']:
            raise RuntimeError('CUDA archive checksum mismatch: ' + component)
    unpack = cache / component
    unpack.mkdir(exist_ok=True)
    with tarfile.open(archive) as handle:
        handle.extractall(unpack, filter='data')
    content = next(unpack.iterdir())
    shutil.copytree(content, prefix, dirs_exist_ok=True, symlinks=True)
    print(component, manifest[component]['version'], flush=True)
print('Local CUDA toolkit:', prefix, flush=True)
