"""Frozen validation using upstream dataset transforms and author-compatible metrics."""
import argparse,hashlib,json,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from surroundocc_protocol import ROOT,DATA,EXPECTED_SAMPLES,confined,confusion_matrix,summarize
from check_surroundocc_data import inspect_data

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=DATA);p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.limit is not None and args.limit<1:p.error('--limit must be positive')
    base=confined(args.data_root);directory=ROOT/'reports/surroundocc';directory.mkdir(parents=True,exist_ok=True)
    preflight=inspect_data(base);(directory/'data-preflight.json').write_text(json.dumps(preflight,indent=2))
    if not preflight['ready']:raise SystemExit('Dataset not ready. See reports/surroundocc/data-preflight.json; no benchmark score was produced.')
    import torch
    from run_gaussianformer3d import build_model
    from surroundocc_dataset import build_dataset
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=True;os.environ['eval']='true'
    started=time.perf_counter();model,config,tensors=build_model(progress=lambda stage:print('MODEL_LOADING '+stage,flush=True))
    if tensors!=941 or any(p.requires_grad for p in model.parameters()):raise RuntimeError('Expected frozen strict 941-tensor checkpoint')
    dataset,collate=build_dataset(base/'nuscenes',base/'nuscenes_cam/nuscenes_infos_gf3d_val.pkl',base/'surroundocc/samples',config)
    if len(dataset)!=EXPECTED_SAMPLES:raise RuntimeError('Incorrect validation split')
    if args.limit:indices=list(range(min(args.limit,len(dataset))));selected=torch.utils.data.Subset(dataset,indices)
    else:selected=dataset
    loader=torch.utils.data.DataLoader(selected,batch_size=1,shuffle=False,num_workers=int(config.val_loader.num_workers),collate_fn=collate,pin_memory=True)
    print(f'VALIDATION_READY {len(selected)} samples; frozen checkpoint tensors {tensors}',flush=True)
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');output=directory/tag;output.mkdir()
    matrix=np.zeros((18,18),np.int64);ignored_matrix=matrix.copy();times=[];peak=0.;seen=0
    def cuda(value):
        if isinstance(value,torch.Tensor):return value.cuda()
        if isinstance(value,dict):return {k:cuda(v) for k,v in value.items()}
        if isinstance(value,list):return [cuda(v) for v in value]
        return value
    with (output/'progress.jsonl').open('w') as stream,torch.inference_mode():
        for batch in loader:
            # Labels remain on CPU; the head only needs the fixed geometric query grid.
            target=batch.pop('occ_label').numpy();batch.pop('occ_cam_mask',None)
            batch=cuda(batch);images=batch.pop('img');points=batch.pop('points');depth=batch.pop('dpt')
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
            result=model(imgs=images,points=points,dpt=depth,metas=batch)
            logits=result['pred_occ'][-1]
            if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite logits')
            prediction=logits.argmax(1).reshape(target.shape).cpu().numpy();torch.cuda.synchronize()
            elapsed=time.perf_counter()-begin;times.append(elapsed);peak=max(peak,torch.cuda.max_memory_allocated()/1e9)
            matrix+=confusion_matrix(prediction,target);ignored_matrix+=confusion_matrix(prediction,target,ignore_zero=True);seen+=1
            row={'processed':seen,'total':len(selected),'forward_argmax_copy_seconds':elapsed};stream.write(json.dumps(row)+'\n');stream.flush()
            if seen%50==0:print(json.dumps(row),flush=True)
            del result,logits,images,points,depth,batch
    with (ROOT/'checkpoints/surroundocc_release.pth').open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    full=seen==EXPECTED_SAMPLES and args.limit is None
    report={'status':'complete_validation' if full else 'partial_validation_diagnostic','full_standard_validation':full,'metric_units':'fractions, 0..1',
        'samples_evaluated':seen,'expected_samples':EXPECTED_SAMPLES,'frozen':True,'strict_tensor_count':tensors,
        'checkpoint_sha256':digest,'preflight':preflight,'author_compatible':summarize(matrix),
        'diagnostic_ignore_gt_zero':summarize(ignored_matrix),'confusion_matrix':matrix.tolist(),
        'protocol':{'source':'GaussianFormer3D/eval.py and misc/metric_util.py','classes':'1..16; empty=17',
            'roi':'Entire 200x200x16 LiDAR grid, 0.5 m, [-50,-50,-5]..[50,50,3]',
            'mask':'Author evaluator applies no camera mask for SurroundOcc and counts GT 0 as nonempty; ignore-GT-zero diagnostic is separate',
            'sweeps':'Unmodified upstream val pipeline, including its configured seeded random selection and timestamp arithmetic',
            'counting':'Each sample counted once; author eval.py counts twice uniformly, which leaves IoU ratios unchanged',
            'preprocessing':'Unmodified author transforms; only paths relocated; narrow BasePoints adapter if mmdet3d is absent',
            'ground_truth_used_for_inference':False},
        'latency':{'scope':'CUDA-synchronized forward, finite-logit check, argmax and CPU label copy; excludes input loading/transfers; first passes included',
                   'samples':len(times),'median_seconds':float(np.median(times)),'p95_seconds':float(np.percentile(times,95)),
                   'median_after_first_2_seconds':float(np.median(times[2:])) if len(times)>2 else None},
        'peak_cuda_allocated_GB':peak,'wall_seconds':time.perf_counter()-started,'device':torch.cuda.get_device_name(),
        'published_reference_only':{'occupied_iou':.433,'semantic_miou':.271,'source':'https://github.com/lunarlab-gatech/GaussianFormer3D'}}
    (output/'results.json').write_text(json.dumps(report,indent=2));(directory/'latest.json').write_text(json.dumps({'report':str(output/'results.json'),'full_standard_validation':full},indent=2))
    print(json.dumps(report['author_compatible'],indent=2));print('RESULTS',output/'results.json')
if __name__=='__main__':main()
