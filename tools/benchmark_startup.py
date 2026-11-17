"""Matched cold-process model startup benchmark; writes inspectable phase timings."""
import time
started=time.perf_counter()
import argparse,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args()
phases={}
def phase(name):
    phases[name]=time.perf_counter()-started
    print(name,round(phases[name],3),flush=True)
import torch
phase('torch imported')
from run_gaussianformer3d import build_model,build_queries
torch.set_num_threads(4);torch.manual_seed(42)
model,config,tensors=build_model(progress=phase)
queries=build_queries(config);torch.cuda.synchronize();phase('ready')
result={'seconds':time.perf_counter()-started,'phases':phases,'strict_tensor_count':tensors,
        'frozen':not any(p.requires_grad for p in model.parameters()),'torch_path':torch.__file__,
        'runtime':os.environ.get('GF3D_FAST_RUNTIME','auto'),'device':torch.cuda.get_device_name(),
        'module_paths':{name:getattr(sys.modules.get(name),'__file__',None) for name in ['torch','mmcv','spconv','numpy','mmengine']}}
args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
