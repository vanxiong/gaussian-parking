#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/linux_env.sh"
cd "$GF3D_ROOT"
python - <<'PY'
import json
from pathlib import Path
import torch
import spconv.pytorch as spconv
with torch.inference_mode():
    x=torch.randn(4,128,device='cuda')
    indices=torch.tensor([[0,1,1,1],[0,2,1,1],[0,1,2,1],[0,1,1,2]],device='cuda',dtype=torch.int32)
    tensor=spconv.SparseConvTensor(x,indices,[4,4,4],1)
    network=spconv.SubMConv3d(128,128,5,padding=2,bias=False).cuda().eval()
    result=network(tensor).features
    assert result.shape==(4,128) and torch.isfinite(result).all()
torch.cuda.synchronize()
report={'spconv_gpu_forward':True,'gpu':torch.cuda.get_device_name(),'shape':list(result.shape),'kernel_size':5}
Path('reports/spconv_probe.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
PY
