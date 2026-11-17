"""Use the author's unmodified dataset transforms with a narrow points adapter."""
import sys,types
from pathlib import Path
import numpy as np
import torch
from surroundocc_protocol import ROOT,resolve_sensor

class EvaluationLiDARPoints:
    """Only tensor/index/new_point/cat operations exercised by the upstream val pipeline."""
    def __init__(self,points,points_dim=5,attribute_dims=None):
        self.tensor=torch.as_tensor(points,dtype=torch.float32).clone()
        if self.tensor.ndim!=2 or self.tensor.shape[1]!=points_dim:raise ValueError('Invalid LiDAR point tensor')
        self.points_dim=points_dim;self.attribute_dims=attribute_dims
    def __len__(self):return len(self.tensor)
    def __getitem__(self,index):
        tensor=self.tensor[index]
        if tensor.ndim==1:tensor=tensor[None]
        return type(self)(tensor,tensor.shape[-1],self.attribute_dims)
    def new_point(self,points):return type(self)(points,self.points_dim,self.attribute_dims)
    @classmethod
    def cat(cls,points):return cls(torch.cat([p.tensor for p in points]),points[0].points_dim,points[0].attribute_dims)

def dataset_api():
    sys.path.insert(0,str(ROOT/'.eval-deps'));sys.path.insert(0,str(ROOT/'GaussianFormer3D'))
    try:from mmdet3d.structures.points import BasePoints
    except ModuleNotFoundError:
        points=types.ModuleType('mmdet3d.structures.points');points.BasePoints=EvaluationLiDARPoints
        def get_points_type(kind):
            if kind!='LIDAR':raise ValueError('Evaluation adapter only supports LiDAR')
            return EvaluationLiDARPoints
        points.get_points_type=get_points_type
        for name in ['mmdet3d','mmdet3d.structures']:
            module=types.ModuleType(name);module.__path__=[];sys.modules[name]=module
        sys.modules['mmdet3d.structures.points']=points
    from dataset.dataset import NuScenesDataset
    from dataset.utils import custom_collate_fn_temporal
    return NuScenesDataset,custom_collate_fn_temporal

def build_dataset(data_root,info_file,occ_root,config):
    from copy import deepcopy
    Dataset,collate=dataset_api()
    pipeline=deepcopy(config.val_dataset_config.pipeline)
    for transform in pipeline:
        if transform['type']=='LoadOccupancySurroundOcc':transform['occ_path']=str(occ_root)
    class RelocatedDataset(Dataset):
        def get_data_info(self,info):
            info=deepcopy(info)
            for record in info['data'].values():
                if 'filename' in record:record['filename']=str(resolve_sensor(record['filename'],data_root))
            if 'lidar_path' in info:info['lidar_path']=str(resolve_sensor(info['lidar_path'],data_root))
            for sweep in info.get('sweeps',[]):sweep['data_path']=str(resolve_sensor(sweep['data_path'],data_root))
            return super().get_data_info(info)
    dataset=RelocatedDataset(data_root=str(data_root),imageset=str(info_file),data_aug_conf=config.val_dataset_config.data_aug_conf,pipeline=pipeline,phase='val')
    return dataset,collate
