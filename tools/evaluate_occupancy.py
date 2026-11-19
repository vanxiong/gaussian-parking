"""Compare raw predictions with independent frozen USD geometry (never feeds inference)."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from occupancy_evaluation import SHAPE,confusion,metrics,voxelize_scene,evaluation_masks

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);args=p.parse_args()
    manifest=json.loads(args.manifest.read_text())
    output=args.manifest.parent;frames=[];totals={};started=time.perf_counter()
    for sample in manifest['samples']:
        capture=Path(sample['capture']);meta=json.loads((capture/'capture.json').read_text())
        geometry=json.loads(Path(sample['geometry']).read_text())
        if not np.allclose(geometry['world_from_ego'],meta['world_from_ego'],atol=1e-5,rtol=0):raise ValueError('Ground truth and sensor pose mismatch')
        if abs(meta['ground_truth_snapshot']['timestamp']-meta['simulation_timestamp_seconds'])>1e-6:raise ValueError('Ground truth and sensor timestamp mismatch')
        raw_path=Path(sample['raw_labels']);raw=np.load(raw_path,allow_pickle=False)
        conf=np.load(Path(sample['confidence']),allow_pickle=False)
        if raw.shape!=SHAPE or conf.shape!=SHAPE or not np.isfinite(conf).all() or raw.min()<0 or raw.max()>17:raise ValueError('Invalid model output')
        transform=np.asarray(meta['world_from_ego'])@np.asarray(meta['lidar']['ego_from_lidar'])
        truth=voxelize_scene(geometry,transform);masks=evaluation_masks(meta,geometry)
        frame={'capture':str(capture),'scene_sha256':geometry['scene_sha256'],
               'prediction_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),
               'raw_noise_label_voxels':int((raw==0).sum()),'metrics':{}}
        for mask_name,mask in masks.items():
            for policy,pred in [('argmax',raw!=17),('confidence_0.35',(raw!=17)&(conf>=.35))]:
                name=mask_name+'/'+policy;counts=confusion(pred,truth,mask)
                frame['metrics'][name]=metrics(counts)
                current=totals.setdefault(name,dict.fromkeys(['tp','fp','fn','tn'],0))
                for key,value in counts.items():current[key]+=value
        np.savez_compressed(output/(capture.name+'-ground-truth.npz'),occupied=truth,**masks,world_from_lidar=transform)
        frames.append(frame);print('EVALUATED',capture.name,flush=True)
    report={'model':'GaussianFormer3D','ground_truth':'USD visible geometry, filled convex solids; ego and guide/proxy geometry excluded',
        'grid':{'shape':SHAPE,'origin':[-50,-50,-5],'voxel_size_m':.5,'axes':'XYZ in current LiDAR frame'},
        'protocol':{'primary':'obstacles_above_ground/argmax','occupied_prediction':'raw argmax != 17, including noise class 0 as nonempty, matching upstream geometric metric',
            'roi':'30 m horizontal radius; within known ground XY bounds; primary voxel-center world height 0.3..3 m; secondary -0.25..3 m',
            'ego_exclusion':'ego-frame XY column |x|<=2.5 m, |y|<=1.4 m',
            'ground_truth_resolution':'Conservative solid/voxel intersection; curved primitives tessellated before voxelization',
            'visibility':'Full known-geometry ROI includes occluded voxels; no sensor visibility mask',
            'aggregation':'Micro confusion counts across frames; correlated samples from one static parking scene',
            'semantic_miou':'Not reported: no independently verified 16-class simulator mapping',
            'confidence_threshold':'0.35 reported only as a secondary fixed visualization policy; no threshold tuning'},
        'aggregate':{name:metrics(value) for name,value in totals.items()},'frames':frames,
        'evaluation_wall_seconds':time.perf_counter()-started,'limitations':manifest['limitations']}
    report['appearance_ablation']=manifest.get('appearance_ablation',{'variant':'simple'})
    report['building_ablation']=manifest.get('building_ablation',{'presence':'present'})
    if report['appearance_ablation']['variant']=='realistic':
        report['ground_truth']='USD filled convex solids; selected cars use shared original simple-car proxies, detailed visuals excluded'
        report['protocol']['appearance_reference']='shared_simple_car_proxy_v1; appearance-only experiment; collision and LiDAR geometry unchanged'
    (output/'accuracy.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['aggregate'],indent=2))

if __name__=='__main__':main()
