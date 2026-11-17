"""Record upstream revisions and local patch evidence without changing checkouts."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
report={}
for name,relative in [('GaussianFormer3D','GaussianFormer3D'),('MMCV','vendor/mmcv'),('DFA3D','vendor/DFA3D')]:
    path=ROOT/relative
    revision=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    diff=subprocess.check_output(['git','-C',str(path),'diff','--no-ext-diff','--binary'])
    (ROOT/f'reports/{name.lower()}-local.patch').write_bytes(diff)
    report[name]={'revision':revision,'local_diff_sha256':hashlib.sha256(diff).hexdigest(),
                  'patch_file':f'reports/{name.lower()}-local.patch'}
(ROOT/'reports/source_versions.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
