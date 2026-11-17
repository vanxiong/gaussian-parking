"""Protocol parity and actual upstream loader tests; synthetic fixtures are not scores."""
import ast,pickle,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from surroundocc_protocol import ROOT,resolve_sensor,sparse_labels,confusion_matrix,summarize

def test_sparse_xyz_class_and_empty_conventions():
    label=sparse_labels(np.array([[1,2,3,4],[4,5,6,0]]))
    assert label[1,2,3]==4 and label[4,5,6]==0 and label[0,0,0]==17

def test_invalid_sparse_data_rejected():
    for values in [[[200,0,0,4]],[[0,0,16,4]],[[0,0,0,17]],[[.5,0,0,4]],[[0,0,0,4],[0,0,0,5]]]:
        with pytest.raises(ValueError):sparse_labels(np.array(values))

def test_path_relocation_and_escape_rejection(tmp_path):
    assert resolve_sensor('/authors/data/nuscenes/samples/CAM_FRONT/test.jpg',tmp_path)==tmp_path/'samples/CAM_FRONT/test.jpg'
    with pytest.raises(ValueError):resolve_sensor('samples/../../escape.bin',tmp_path)

def test_ignored_class_diagnostic_differs_explicitly():
    pred=np.array([17,4,17]);gt=np.array([0,4,17])
    assert summarize(confusion_matrix(pred,gt))['occupied_iou']==.5
    assert summarize(confusion_matrix(pred,gt,ignore_zero=True))['occupied_iou']==1

def test_exact_metric_parity_with_author_source():
    source=ROOT/'GaussianFormer3D/misc/metric_util.py'
    definition=next(node for node in ast.parse(source.read_text()).body if isinstance(node,ast.ClassDef) and node.name=='MeanIoU')
    namespace={'np':np,'torch':torch,'dist':torch.distributed,'logger':SimpleNamespace(info=lambda *a:None)}
    exec(compile(ast.Module(body=[definition],type_ignores=[]),str(source),'exec'),namespace)
    original=namespace['MeanIoU'](list(range(1,17)),17,[str(i) for i in range(16)],training=False)
    original.total_seen=torch.zeros(17);original.total_positive=torch.zeros(17);original.total_correct=torch.zeros(17)
    pred=np.array([4,4,0,17,15,17]);gt=np.array([4,17,0,4,17,11])
    original._after_step(torch.tensor(pred),torch.tensor(gt));miou,iou=original._after_epoch()
    ours=summarize(confusion_matrix(pred,gt))
    np.testing.assert_allclose([ours['semantic_miou']*100,ours['occupied_iou']*100],[miou,float(iou)],rtol=1e-6)

def test_upstream_validation_loader_end_to_end(tmp_path):
    from PIL import Image
    from mmengine import Config
    from surroundocc_dataset import build_dataset
    root=tmp_path/'nuscenes';occ=tmp_path/'occ';occ.mkdir();(root/'depth_gt').mkdir(parents=True)
    cameras=['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
    sensor={};calib={'rotation':[1,0,0,0],'translation':[0,0,0]};pose=dict(calib)
    for name in cameras:
        rel=f'samples/{name}/frame.jpg';path=root/rel;path.parent.mkdir(parents=True)
        Image.new('RGB',(60,36),(100,50,20)).save(path,quality=100,subsampling=0)
        np.array([[10,10,5]],np.float32).tofile(root/'depth_gt'/(path.name+'.bin'))
        sensor[name]={'filename':rel,'calib':{**calib,'camera_intrinsic':[[10,0,16],[0,10,16],[0,0,1]]},'pose':pose}
    lidar=root/'samples/LIDAR_TOP/frame.pcd.bin';lidar.parent.mkdir(parents=True)
    np.array([[2,2,0,50,5],[3,3,0,100,6]],np.float32).tofile(lidar)
    sensor['LIDAR_TOP']={'filename':'samples/LIDAR_TOP/frame.pcd.bin','calib':calib,'pose':pose,'token':'lidar-token'}
    info={'token':'synthetic-token','timestamp':1_000_000,'data':sensor,'occ_path':'unused','lidar_path':str(lidar),'sweeps':[]}
    annotations=tmp_path/'fixture.pkl';annotations.write_bytes(pickle.dumps({'infos':{'synthetic-scene':[info]},'metadata':[('synthetic-scene',0)]}))
    np.save(occ/'frame.pcd.bin.npy',np.array([[100,100,10,4],[101,100,10,0]],np.int64))
    config=Config.fromfile(str(ROOT/'GaussianFormer3D/config/nuscenes_surroundocc_gs25600.py'))
    dataset,collate=build_dataset(root,annotations,occ,config);item=dataset[0];batch=collate([item])
    assert tuple(batch['img'].shape)==(1,6,3,64,64)
    assert tuple(batch['dpt'].shape)==(1,6,1,64,64) and batch['dpt'][0,0,0,10,10]==5
    np.testing.assert_allclose(batch['img'][0,0,:,10,10].numpy(),(np.array([100,50,20])-[123.675,116.28,103.53])/[58.395,57.12,57.375],atol=.025)
    assert torch.all(batch['img'][0,0,:,-1,-1]==0)
    assert tuple(batch['points'].shape)==(1,22,5) and torch.all(batch['points'][:,:,4]==0)
    assert item['occ_label'][100,100,10]==4 and not item['occ_cam_mask'][101,100,10]
    np.testing.assert_allclose(item['occ_xyz'][100,100,10],[.25,.25,.25])

def test_official_validation_scene_list():
    import json
    info=json.loads((ROOT/'configs/surroundocc_val_scenes.json').read_text())
    assert len(set(info['scene_names']))==150 and info['expected_samples']==6019

def test_author_sweep_transform_and_timestamp_arithmetic(tmp_path):
    from surroundocc_dataset import dataset_api,EvaluationLiDARPoints
    dataset_api()
    from dataset.transform_3d import LoadPointsFromMultiSweepsLiDAR
    path=tmp_path/'sweep.bin';np.array([[2,3,4,50,99]],np.float32).tofile(path)
    data={'points':EvaluationLiDARPoints([[2,2,0,10,7]]),'timestamp':1.,
          'sweeps':[{'data_path':str(path),'timestamp':500_000,
                     'sensor2lidar_rotation':np.array([[0,-1,0],[1,0,0],[0,0,1.]]),
                     'sensor2lidar_translation':np.array([1,0,0])}]}
    result=LoadPointsFromMultiSweepsLiDAR(sweeps_num=10,use_dim=5,pad_empty_sweeps=True,remove_close=True)(data)['points'].tensor.numpy()
    np.testing.assert_allclose(result[1,:4],[-2,2,4,50])
    np.testing.assert_allclose(result[1,4],1e-6-.5,atol=1e-7)
