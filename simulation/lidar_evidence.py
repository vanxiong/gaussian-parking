"""Build local clearance from measured LiDAR; learned occupancy supplies soft cost."""
import json
import math
from pathlib import Path
import cv2
import numpy as np


def update_evidence(evidence,capture,labels,confidence,reference,footprint,min_solid_height=.30):
    capture=Path(capture)
    metadata=json.loads((capture/'capture.json').read_text())
    lidar=metadata['lidar']
    scans=np.load(capture/lidar['points'],allow_pickle=False)
    current=scans[np.isclose(scans[:,4],0)]
    if len(current)<100:raise ValueError('Insufficient current LiDAR hits for clearance')
    ego=np.asarray(metadata['world_from_ego'])
    world_from_lidar=ego@np.asarray(lidar['ego_from_lidar'])
    metrics=update_lidar(evidence,current[:,:3],ego,world_from_lidar,footprint,min_solid_height)
    indices=np.argwhere((labels>0)&~np.isin(labels,[11,12,13,14,17])&(confidence>=.35))
    local_xyz=(indices+.5)*.5+[-50.,-50.,-5.]
    predicted=local_xyz@reference[:3,:3].T+reference[:3,3]
    metrics['gaussian_obstacle_voxels']=update_semantics(evidence,predicted)
    metrics['semantic_cost_source']='frozen GaussianFormer3D'
    return metrics


def update_lidar(evidence,points,ego,world_from_lidar,footprint,min_solid_height=.30):
    """Update hard clearance from memory-backed current-scan points."""
    if len(points)<100:raise ValueError('Insufficient current LiDAR hits for clearance')
    current=points
    xyz=points[:,:3]@world_from_lidar[:3,:3].T+world_from_lidar[:3,3]
    local=np.c_[xyz,np.ones(len(xyz))]@np.linalg.inv(ego).T
    own=(np.abs(local[:,0])<footprint[0]/2+.25)&(np.abs(local[:,1])<footprint[1]/2+.35)
    solid=xyz[(~own)&(xyz[:,2]>=min_solid_height)&(xyz[:,2]<2.5)]
    floor=np.zeros_like(evidence.free,np.uint8)
    obstacle=np.zeros_like(floor)
    start=np.floor((world_from_lidar[:2,3]-evidence.origin)/evidence.res).astype(int)
    # Only the current scan clears old evidence; historical returns never freeze moving obstacles.
    visible=xyz[(xyz[:,2]>=-.10)&(xyz[:,2]<2.5)]
    ends=np.floor((visible[:,:2]-evidence.origin)/evidence.res).astype(int)
    for end in np.unique(ends,axis=0):
        cv2.line(floor,tuple(start),tuple(end),1,1)
    ground=xyz[(xyz[:,2]>=-.10)&(xyz[:,2]<.14)]
    ij=evidence.cells(ground[:,:2]);floor[ij[:,1],ij[:,0]]=1
    ij=evidence.cells(solid[:,:2]);obstacle[ij[:,1],ij[:,0]]=1
    floor=cv2.dilate(floor,np.ones((5,5),np.uint8)).astype(bool)
    obstacle=cv2.dilate(obstacle,np.ones((3,3),np.uint8)).astype(bool)
    evidence.free|=floor
    evidence.obstacle[floor]=False
    evidence.obstacle|=obstacle
    evidence.free[evidence.obstacle]=False
    h=math.atan2(ego[1,0],ego[0,0]);center=ego[:2,3]
    yy,xx=np.mgrid[:evidence.free.shape[0],:evidence.free.shape[1]]
    xy=np.stack([xx,yy],-1)*evidence.res+evidence.origin+evidence.res/2-center
    ex=xy[:,:,0]*math.cos(h)+xy[:,:,1]*math.sin(h)
    ey=-xy[:,:,0]*math.sin(h)+xy[:,:,1]*math.cos(h)
    own_cells=(abs(ex)<footprint[0]/2+.05)&(abs(ey)<footprint[1]/2+.05)
    evidence.free[own_cells]=True;evidence.obstacle[own_cells]=False
    return {'lidar_current_points':len(current),'lidar_obstacle_cells':int(evidence.obstacle.sum()),
            'observed_free_cells':int(evidence.free.sum()),
            'clearance_source':'current simulated LiDAR; no rendered camera depth'}


def update_semantics(evidence,predicted):
    """predicted contains model obstacle-class voxel centers already in map coordinates."""
    selected=(predicted[:,2]>.14)&(predicted[:,2]<2.5)
    evidence.semantic*=.8
    ij=evidence.cells(predicted[selected,:2]);evidence.semantic[ij[:,1],ij[:,0]]=1.
    evidence.semantic=cv2.GaussianBlur(evidence.semantic,(11,11),1.8)
    return int(selected.sum())
