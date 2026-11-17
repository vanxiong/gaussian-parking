"""Frozen-model latency and prediction benchmark. Ground truth is never opened here."""
import time
PROCESS_STARTED=time.perf_counter()
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from run_gaussianformer3d import ROOT,build_model,build_queries,infer_arrays
from prepare_capture import prepare

def distribution(values):
    return {'n':len(values),'mean_seconds':float(np.mean(values)),'median_seconds':float(np.median(values)),
            'p95_seconds':float(np.percentile(values,95)),'min_seconds':float(min(values)),'max_seconds':float(max(values))}

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);args=p.parse_args()
    manifest=json.loads(args.manifest.read_text());output=args.manifest.parent
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(42);np.random.seed(42)
    model,config,tensors=build_model();queries=build_queries(config);torch.cuda.synchronize()
    startup=time.perf_counter()-PROCESS_STARTED
    arrays=prepare(Path(manifest['samples'][0]['capture']))
    warmup=[]
    for _ in range(manifest['warmup']):
        _,_,stats=infer_arrays(model,config,arrays,queries);warmup.append(stats['inference_seconds'])
    timings=[]
    for repeat in range(manifest['repeats']):
        for sample in manifest['samples']:
            capture=Path(sample['capture']);start=time.perf_counter();arrays=prepare(capture);prepared=time.perf_counter()
            labels,confidence,stats=infer_arrays(model,config,arrays,queries)
            end=time.perf_counter()
            timings.append({'capture':str(capture),'repeat':repeat,'preprocess_seconds':prepared-start,
                'pipeline_seconds':end-start,**stats})
            if repeat==0:
                destination=output/'predictions'/capture.name;destination.mkdir(parents=True,exist_ok=True)
                np.save(destination/'raw_labels.npy',labels);np.save(destination/'confidence.npy',confidence)
                sample['raw_labels']=str(destination/'raw_labels.npy');sample['confidence']=str(destination/'confidence.npy')
            print('INFERRED',capture.name,repeat,round(stats['inference_seconds'],4),flush=True)
    frozen=not any(p.requires_grad for p in model.parameters())
    if tensors!=941 or not frozen:raise RuntimeError('Expected strict frozen pretrained model')
    with (ROOT/'checkpoints/surroundocc_release.pth').open('rb') as stream:checkpoint=hashlib.file_digest(stream,'sha256').hexdigest()
    report={'model':'GaussianFormer3D','frozen':frozen,'strict_tensor_count':tensors,'checkpoint_sha256':checkpoint,
        'device':torch.cuda.get_device_name(),'pytorch':torch.__version__,'startup_seconds':startup,
        'warmup_forwards':manifest['warmup'],'warmup_inference_seconds':warmup,'batch_size':1,
        'timed_forwards':len(timings),'unique_frames':len(manifest['samples']),
        'inference':distribution([r['inference_seconds'] for r in timings]),
        'preprocessing':distribution([r['preprocess_seconds'] for r in timings]),
        'pipeline':distribution([r['pipeline_seconds'] for r in timings]),
        'peak_cuda_allocated_GB':max(r['peak_cuda_allocated_GB'] for r in timings),
        'scopes':{'inference':'CUDA-synchronized forward + softmax/argmax; excludes host/device input transfers and final CPU output copy',
                  'pipeline':'saved sensor input read + RGB/LiDAR preprocessing + input CUDA transfer + inference + CPU output copy; excludes output saving, simulator rendering, startup, and ground truth',
                  'throughput':'inverse mean latency describes sequential batch-1 replay, not live simulator FPS'},
        'timings':timings,'ground_truth_used_for_inference':False}
    (output/'latency.json').write_text(json.dumps(report,indent=2));args.manifest.write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
