import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from occupancy_evaluation import voxelize_object,voxelize_scene,confusion,metrics,centers,SHAPE,triangle_intersections,evaluation_masks

def cube(lo,hi):return np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])

def test_small_object_crosses_voxel_face():
    # Cell centers alone miss this thin object; exact intersection must retain it.
    voxels=voxelize_object(cube([.49,.1,.1],[.51,.2,.2]))
    assert voxels[100,100,10] and voxels[101,100,10]
    assert voxels.sum()==2

def test_solid_interior_is_occupied():
    voxels=voxelize_object(cube([.1,.1,.1],[1.9,1.9,1.9]))
    assert voxels[102,102,12] and voxels.sum()==64

def test_triangle_box_sat_rejects_aabb_false_hit():
    triangle=np.array([[0,0,0],[1,0,0],[0,1,0]])
    assert triangle_intersections(triangle,np.array([[.25,.25,0],[.95,.95,0]])).tolist()==[True,False]

def test_transform_rotates_translates_ground_truth_into_lidar_frame():
    angle=np.pi/2;transform=np.array([[0,-1,0,4],[1,0,0,5],[0,0,1,1],[0,0,0,1.]])
    local=cube([.1,.1,.1],[.4,.4,.4]);world=local@transform[:3,:3].T+transform[:3,3]
    truth=voxelize_scene({'objects':[{'points_world':world.tolist()}]},transform)
    assert truth.sum()==1 and truth[100,100,10]

def test_confusion_counts_and_iou():
    prediction=np.array([1,1,1,0,0],bool);truth=np.array([1,1,0,1,0],bool)
    score=metrics(confusion(prediction,truth,np.ones(5,bool)))
    assert score['tp']==2 and score['fp']==1 and score['fn']==1 and score['tn']==1
    assert score['occupied_iou']==.5 and score['f1']==2/3

def test_masked_voxels_do_not_affect_score():
    assert confusion(np.array([1,0],bool),np.array([0,1],bool),np.array([0,1],bool))=={'tp':0,'fp':0,'fn':1,'tn':0}

def test_undefined_metrics_are_explicit():
    assert metrics(dict.fromkeys(['tp','fp','fn','tn'],0))['occupied_iou'] is None

def test_perfect_prediction_and_empty_baseline():
    truth=np.array([True,False,True]);mask=np.ones(3,bool)
    assert metrics(confusion(truth,truth,mask))['occupied_iou']==1.
    empty=metrics(confusion(np.zeros(3,bool),truth,mask))
    assert empty['occupied_iou']==0 and empty['recall']==0 and empty['precision'] is None

def test_masks_exclude_ego_below_ground_and_unknown_bounds():
    meta={'world_from_ego':np.eye(4).tolist(),'lidar':{'ego_from_lidar':np.eye(4).tolist()}}
    mask=evaluation_masks(meta,{'known_xy_bounds':[[-10,-10],[10,10]]})['obstacles_above_ground']
    assert not mask[100,100,11] # ego footprint
    assert not mask[110,100,9] # below ground
    assert mask[110,100,11] # known free/occupied space outside ego
    assert not mask[130,100,11] # outside known scene
