"""Controlled input probes on a saved capture; diagnostic only, no model changes."""
import argparse,json
from pathlib import Path
import numpy as np
import torch,cv2
from run_gaussianformer3d import build_model,build_queries,infer_arrays
from prepare_capture import prepare,prepare_arrays
from occupancy_evaluation import confusion,metrics
from PIL import Image

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('--repeatability-only',action='store_true');a=p.parse_args()
    manifest=json.loads(a.manifest.read_text());s=manifest['samples'][0];capture=Path(s['capture']);out=a.manifest.parent/'diagnostics'
    meta=json.loads((capture/'capture.json').read_text());points=np.load(capture/meta['lidar']['points'])
    rgb={c['name']:np.array(Image.open(capture/c['image']).convert('RGB')) for c in meta['cameras']}
    gt=np.load(a.manifest.parent/(capture.name+'-ground-truth.npz'));truth=gt['occupied'];mask=gt['obstacles_above_ground']
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.manual_seed(42);np.random.seed(42);model,config,_=build_model();queries=build_queries(config)
    result={'warning':'Exploratory sensitivity probes on one capture; changed intensities are synthetic assumptions, not calibrated reflectance. Upsampling cannot restore image detail. No setting is selected for deployment.','runs':[]}
    baseline=prepare(capture)
    if a.repeatability_only:
        rows=[];reference=None
        for repeat in range(3):
            torch.manual_seed(42);np.random.seed(42)
            raw,_,_=infer_arrays(model,config,baseline,queries)
            if reference is None:reference=raw.copy()
            rows.append({'repeat':repeat,'seed':42,**metrics(confusion(raw!=17,truth,mask)),
                         'all_voxel_label_agreement_to_first':float((raw==reference).mean()),
                         'roi_occupancy_agreement_to_first':float(((raw!=17)[mask]==(reference!=17)[mask]).mean())})
        (out/'repeatability.json').write_text(json.dumps({'runs':rows,'warning':'Identical arrays and reset seed; residual differences indicate nondeterministic runtime behavior.'},indent=2))
        print(json.dumps(rows,indent=2),flush=True);return
    variants={'baseline':baseline}
    for value in [0.,64.,128.]:
        changed=points.copy();changed[:,3]=value
        variants[f'intensity_{int(value)}']=prepare_arrays(meta,rgb,changed)
    variants['current_sweep_only']=prepare_arrays(meta,rgb,points[np.isclose(points[:,4],0)])
    larger=json.loads(json.dumps(meta));images={}
    for c in larger['cameras']:
        w,h=c['resolution'];nw,nh=1600,900
        images[c['name']]=cv2.resize(rgb[c['name']],(nw,nh),interpolation=cv2.INTER_LINEAR)
        k=np.asarray(c['intrinsic']);k[0]*=nw/w;k[1]*=nh/h;c['intrinsic']=k.tolist();c['resolution']=[nw,nh]
    variants['upsampled_1600x900']=prepare_arrays(larger,images,points)
    for name,arrays in variants.items():
        for seed in [42,43,44]:
            torch.manual_seed(seed);np.random.seed(seed)
            raw,confidence,stats=infer_arrays(model,config,arrays,queries)
            score=metrics(confusion(raw!=17,truth,mask));record={'variant':name,'seed':seed,**score,**stats}
            result['runs'].append(record)
            np.save(out/f'probe-{name}-{seed}.npy',raw)
            (out/'input-probes.json').write_text(json.dumps(result,indent=2))
            print(name,seed,score['occupied_iou'],score['precision'],score['recall'],flush=True)
    # Same seed and arrays twice distinguishes random anchor selection from kernel nondeterminism.
    torch.manual_seed(42);np.random.seed(42)
    raw,_,_=infer_arrays(model,config,baseline,queries)
    result['same_seed_voxel_agreement']=float((raw==np.load(out/'probe-baseline-42.npy')).mean())
    (out/'input-probes.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
