"""Read-only diagnostics of one evaluated capture; never tunes the primary score."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from occupancy_evaluation import confusion,metrics,ORIGIN,RESOLUTION,centers,voxelize_scene
from prepare_capture import prepare,ORDER

def shifted(array,offset):
    out=np.zeros_like(array)
    src=[];dst=[]
    for n,d in zip(array.shape,offset):
        src.append(slice(max(0,-d),min(n,n-d)))
        dst.append(slice(max(0,d),min(n,n+d)))
    out[tuple(dst)]=array[tuple(src)]
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);a=p.parse_args()
    manifest=json.loads(a.manifest.read_text());sample=manifest['samples'][0]
    capture=Path(sample['capture']);out=a.manifest.parent/'diagnostics';out.mkdir(exist_ok=True)
    meta=json.loads((capture/'capture.json').read_text());geometry=json.loads(Path(sample['geometry']).read_text())
    gt=np.load(a.manifest.parent/(capture.name+'-ground-truth.npz'))
    truth=gt['occupied'];mask=gt['obstacles_above_ground'];raw=np.load(sample['raw_labels']);pred=raw!=17
    score=lambda x:metrics(confusion(x,truth,mask))
    result={'source_manifest':str(a.manifest),'baseline':score(pred),'alignment_tests':[],
            'warning':'Post-hoc diagnostic searches use ground truth, are optimistic, and do not replace the primary evaluation.'}
    expected=json.loads((a.manifest.parent/'accuracy.json').read_text())['frames'][0]['metrics']['obstacles_above_ground/argmax']
    if result['baseline']!=expected:raise ValueError('Diagnostic baseline differs from recorded frame metrics')
    for swap in [False,True]:
        for fx in [False,True]:
            for fy in [False,True]:
                v=pred.swapaxes(0,1) if swap else pred
                if fx:v=v[::-1]
                if fy:v=v[:,::-1]
                result['alignment_tests'].append({'swap_xy':swap,'flip_x':fx,'flip_y':fy,**score(v)})
    result['alignment_tests'].append({'flip_z':True,**score(pred[:,:,::-1])})
    # Local offsets up to 2 m on each axis, with no wrapping.
    offsets=[]
    for x in range(-4,5):
        for y in range(-4,5):
            for z in range(-4,5):
                s=score(shifted(pred,(x,y,z)))
                offsets.append({'offset_m':[x*.5,y*.5,z*.5],**s})
    result['best_local_translations']=sorted(offsets,key=lambda r:r['occupied_iou'],reverse=True)[:8]
    result['best_xy_translation']=max((r for r in offsets if r['offset_m'][2]==0),key=lambda r:r['occupied_iou'])
    result['best_z_translation']=max((r for r in offsets if r['offset_m'][:2]==[0.,0.]),key=lambda r:r['occupied_iou'])
    best=np.rint(np.array(result['best_local_translations'][0]['offset_m'])/.5).astype(int)
    result['translation_roi_effect']={'original_occupied_predictions':int((pred&mask).sum()),
        'original_predictions_shifted_out_of_roi':int((pred&mask).sum()- (shifted(pred&mask,best)&mask).sum()),
        'original_false_positives_shifted_out_of_roi':int((pred&~truth&mask).sum()-(shifted(pred&~truth&mask,best)&mask).sum())}
    result['yaw_tests']=[{'yaw_deg':angle,**score(ndimage.rotate(pred,angle,axes=(0,1),reshape=False,order=0,prefilter=False))} for angle in [-10,-5,-2,0,2,5,10]]
    xyz=centers(np.indices(pred.shape).reshape(3,-1).T).reshape(*pred.shape,3)
    world=xyz@gt['world_from_lidar'][:3,:3].T+gt['world_from_lidar'][:3,3]
    result['range_bands']=[{'range_m':[lo,hi],**metrics(confusion(pred,truth,mask&(np.linalg.norm(xyz[:,:,:,:2],axis=-1)>=lo)&(np.linalg.norm(xyz[:,:,:,:2],axis=-1)<hi)))} for lo,hi in [(0,10),(10,20),(20,30.001)]]
    fp=pred&~truth&mask;distance=ndimage.distance_transform_edt(~(truth&mask))*.5
    result['false_positive_distance']={f'beyond_{d}_m':int((fp&(distance>d)).sum()) for d in [.5,1.,2.,3.]}
    result['false_positive_distance_reference']='Distance between voxel centers to occupied ground-truth cells in the primary above-ground ROI; ground excluded.'
    result['false_positive_classes']={str(c):int((fp&(raw==c)).sum()) for c in range(18) if (fp&(raw==c)).any()}
    result['false_positive_height']=[{'height_m':[lo,hi],'count':int((fp&(world[:,:,:,2]>=lo)&(world[:,:,:,2]<hi)).sum())} for lo,hi in [(.3,1),(1,2),(2,3.001)]]
    groups={}
    for obj in geometry['objects']:
        key=obj['path'].split('/')[2]
        # Keep the important large structure separate; other objects remain one bucket.
        group='building' if key=='Building' else 'other_geometry'
        groups.setdefault(group,[]).append(obj)
    result['ground_truth_group_recall']={}
    claimed=np.zeros_like(mask)
    for group in ['building','other_geometry']:
        occupied=voxelize_scene({'objects':groups.get(group,[])},gt['world_from_lidar'])&mask&~claimed
        claimed|=occupied
        result['ground_truth_group_recall'][group]={'occupied_voxels':int(occupied.sum()),'correct':int((occupied&pred).sum()),'missed':int((occupied&~pred).sum()),'recall':float((occupied&pred).sum()/occupied.sum()) if occupied.any() else None}
    if not np.array_equal(claimed,truth&mask):raise ValueError('Geometry groups do not reconcile to primary ground truth')
    if sum(result['false_positive_classes'].values())!=result['baseline']['fp']:raise ValueError('Class breakdown does not reconcile')
    points=np.load(capture/meta['lidar']['points']);points=points[np.isclose(points[:,4],0),:3]
    pw=points@gt['world_from_lidar'][:3,:3].T+gt['world_from_lidar'][:3,3]
    # Plane residual is zero on each convex solid's boundary; it is not Euclidean
    # distance outside corners. This checks LiDAR/geometry alignment independently of predictions.
    residual=np.full(len(pw),np.inf)
    for obj in geometry['objects']:
        vertices=np.asarray(obj['points_world']);eq=np.unique(np.round(ConvexHull(vertices).equations,9),axis=0)
        nearby=((pw>=vertices.min(0)-.3)&(pw<=vertices.max(0)+.3)).all(1)
        if nearby.any():residual[nearby]=np.minimum(residual[nearby],np.abs((pw[nearby]@eq[:,:3].T+eq[:,3]).max(1)))
    result['lidar_geometry_surface_check']={'current_returns':len(points),'within_5cm':float((residual<=.05).mean()),'within_20cm':float((residual<=.2).mean()),'median_plane_residual_m':float(np.median(residual)), 'method':'Nearest absolute max convex-hull-plane residual; candidate bounds expanded 0.3 m; rendered shapes may differ from colliders.'}
    free=np.zeros_like(mask)
    ranges=np.linalg.norm(points,axis=1)
    # Sample only the interior of measured rays, with a 0.5 m margin before hits.
    for step in np.arange(.5,float(ranges.max()),.25):
        active=ranges>step+.5
        samples=points[active]*(step/ranges[active,None])
        idx=np.floor((samples-ORIGIN)/RESOLUTION).astype(int)
        valid=((idx>=0)&(idx<np.array(mask.shape))).all(1);idx=idx[valid]
        free[tuple(idx.T)]=True
    # Ray evidence covers parts of cells, so remove cells intersecting known solid geometry.
    free&=mask&~truth
    result['lidar_free_ray_evidence']={'sample_spacing_m':.25,'margin_before_hit_m':.5,'known_empty_cells_crossed_by_rays':int(free.sum()),'false_positives_in_these_cells':int((free&pred).sum()),'warning':'A ray traverses part of a cell; known solid-intersecting cells are excluded. This is supporting evidence, not a replacement visibility protocol.'}
    arrays=prepare(capture);fig,axes=plt.subplots(2,3,figsize=(16,7),layout='constrained')
    result['camera_projection_checks']=[]
    for i,(name,ax) in enumerate(zip(ORDER,axes.flat)):
        cam=next(c for c in meta['cameras'] if c['name']==name);im=np.asarray(Image.open(capture/cam['image']));h,w=im.shape[:2]
        q=np.c_[points,np.ones(len(points))]@arrays['projection_mat'][i].T
        uv=q[:,:2]/np.maximum(q[:,2:3],1e-8);valid=(q[:,2]>.1)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        ax.imshow(im);ax.scatter(uv[valid,0],uv[valid,1],s=.7,c=q[valid,2],cmap='turbo',vmin=0,vmax=40,alpha=.7)
        ax.set(title=f'{name}: {valid.sum()} projected returns',xlim=(0,w),ylim=(h,0));ax.axis('off')
        expected=np.asarray(cam['projection_from_ego'])@np.asarray(meta['lidar']['ego_from_lidar'])
        result['camera_projection_checks'].append({'camera':name,'max_projection_coefficient_error':float(np.abs(expected-arrays['projection_mat'][i,:3]).max()),'visible_returns':int(valid.sum())})
    fig.suptitle('Current LiDAR returns projected through model input calibration (color = optical depth)')
    fig.savefig(out/'lidar-camera-projections.png',dpi=150);plt.close(fig)
    result['prediction_replay_agreement']=float((raw==np.load(capture/'raw_labels.npy')).mean())
    (out/'diagnostics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
