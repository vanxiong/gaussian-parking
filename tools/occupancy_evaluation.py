"""Geometry-only voxelization and occupancy scoring on the model's XYZ grid."""
import numpy as np
from scipy.spatial import ConvexHull

SHAPE=(200,200,16)
ORIGIN=np.array([-50.,-50.,-5.])
RESOLUTION=.5

def centers(indices):return ORIGIN+(indices+.5)*RESOLUTION

def candidates(points):
    lo=np.maximum(np.floor((points.min(0)-ORIGIN-1e-8)/RESOLUTION).astype(int),0)
    hi=np.minimum(np.floor((points.max(0)-ORIGIN+1e-8)/RESOLUTION).astype(int),np.array(SHAPE)-1)
    if (lo>hi).any():return np.empty((0,3),int)
    return np.stack(np.meshgrid(*[np.arange(a,b+1) for a,b in zip(lo,hi)],indexing='ij'),-1).reshape(-1,3)

def triangle_intersections(triangle,xyz):
    """Separating-axis test: triangle versus axis-aligned half-meter voxels."""
    edges=np.roll(triangle,-1,axis=0)-triangle
    axes=[*np.eye(3),np.cross(edges[0],edges[1])]
    axes.extend(np.cross(edge,basis) for edge in edges for basis in np.eye(3))
    keep=np.ones(len(xyz),bool)
    for axis in axes:
        if np.linalg.norm(axis)<1e-12:continue
        projection=triangle@axis;mid=xyz@axis;radius=RESOLUTION/2*np.abs(axis).sum()
        keep &= (mid+radius>=projection.min()-1e-8)&(mid-radius<=projection.max()+1e-8)
    return keep

def voxelize_object(points):
    """Filled convex solid: interior centers plus every intersecting surface cell."""
    points=np.unique(points,axis=0);hull=ConvexHull(points)
    idx=candidates(points);xyz=centers(idx)
    occupied=np.zeros(SHAPE,bool)
    if not len(idx):return occupied
    inside=np.ones(len(xyz),bool)
    for plane in hull.equations:inside &= xyz@plane[:3]+plane[3]<=1e-8
    occupied[tuple(idx[inside].T)]=True
    for face in hull.simplices:
        triangle=points[face];cells=candidates(triangle)
        if not len(cells):continue
        hit=triangle_intersections(triangle,centers(cells))
        occupied[tuple(cells[hit].T)]=True
    return occupied

def voxelize_scene(geometry,world_from_lidar):
    inverse=np.linalg.inv(world_from_lidar);occupied=np.zeros(SHAPE,bool)
    for obj in geometry['objects']:
        points=np.asarray(obj['points_world'])
        local=points@inverse[:3,:3].T+inverse[:3,3]
        if (local.max(0)<ORIGIN).any() or (local.min(0)>ORIGIN+np.array(SHAPE)*RESOLUTION).any():continue
        occupied |= voxelize_object(local)
    return occupied

def evaluation_masks(metadata,geometry):
    idx=np.indices(SHAPE).reshape(3,-1).T;local=centers(idx)
    ego=np.asarray(metadata['world_from_ego']);transform=ego@np.asarray(metadata['lidar']['ego_from_lidar'])
    world=local@transform[:3,:3].T+transform[:3,3]
    lo,hi=np.asarray(geometry['known_xy_bounds']);xy=world[:,:2]
    known=((xy>=lo)&(xy<=hi)).all(1)&(np.linalg.norm(local[:,:2],axis=1)<=30.)
    ego_from_world=np.linalg.inv(ego);body=world@ego_from_world[:3,:3].T+ego_from_world[:3,3]
    known &= ~((np.abs(body[:,0])<=2.5)&(np.abs(body[:,1])<=1.4))
    return {name:(known&(world[:,2]>=low)&(world[:,2]<=3.)).reshape(SHAPE)
            for name,low in [('obstacles_above_ground',.3),('including_ground',-.25)]}

def confusion(prediction,truth,mask):
    p=prediction[mask];t=truth[mask]
    return {name:int(np.count_nonzero(value)) for name,value in {
        'tp':p&t,'fp':p&~t,'fn':~p&t,'tn':~p&~t}.items()}

def metrics(counts):
    tp,fp,fn,tn=(counts[k] for k in ['tp','fp','fn','tn'])
    ratio=lambda a,b:float(a/b) if b else None
    return {**counts,'occupied_iou':ratio(tp,tp+fp+fn),'precision':ratio(tp,tp+fp),
            'recall':ratio(tp,tp+fn),'f1':ratio(2*tp,2*tp+fp+fn),
            'voxel_accuracy':ratio(tp+tn,tp+fp+fn+tn),'evaluated_voxels':tp+fp+fn+tn,
            'ground_truth_occupied_voxels':tp+fn,'all_empty_baseline_accuracy':ratio(tn+fp,tp+fp+fn+tn)}
