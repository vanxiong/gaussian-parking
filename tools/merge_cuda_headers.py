"""Expose library headers shipped with PyTorch's NVIDIA wheels to local nvcc."""
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
prefix=ROOT/'.toolchains/cuda/include'
site=Path(sys.prefix)/'lib/python3.12/site-packages/nvidia'
copied=0
for directory in site.glob('*/include'):
    for source in directory.rglob('*'):
        if source.is_file() and source.suffix not in ['.py','.pyc']:
            target=prefix/source.relative_to(directory)
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
                copied+=1
if not (prefix/'cusparse.h').is_file() or not (prefix/'cublas_v2.h').is_file():
    raise RuntimeError('Missing CUDA linear algebra headers')
print('Added NVIDIA wheel library headers:',copied,flush=True)
