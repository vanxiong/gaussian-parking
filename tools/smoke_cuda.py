"""Execute required CUDA operators and record concrete model-readiness evidence."""
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'GaussianFormer3D'))
from mmcv.ops import modulated_deform_conv2d, Voxelization
import spconv.pytorch as spconv
from dfa3D import _ext as dfa_ext
from model.head.localagg.local_aggregate import LocalAggregator
from model.lifter.gaussian_lifter import GaussianLifterLiDAR
from model.utils.safe_ops import safe_sigmoid, point_cloud_map
from model.compat import HardSimpleVFE

torch.manual_seed(17)
with torch.inference_mode():
    image=torch.randn(1,2,8,8,device='cuda')
    weight=torch.randn(3,2,3,3,device='cuda')
    bias=torch.randn(3,device='cuda')
    offset=torch.zeros(1,18,8,8,device='cuda')
    mask=torch.ones(1,9,8,8,device='cuda')
    actual=modulated_deform_conv2d(image,offset,mask,weight,bias,1,1,1,1,1)
    expected=F.conv2d(image,weight,bias,padding=1)
    torch.testing.assert_close(actual,expected,atol=1e-4,rtol=1e-4)
    voxelize=Voxelization([1,1,1],[0,0,0,4,4,4],max_num_points=5,max_voxels=20)
    points=torch.tensor([[.1,.1,.1,10,0],[.2,.2,.2,20,0],[2.1,2.1,2.1,30,0]],device='cuda')
    voxels,coordinates,counts=voxelize(points)
    assert sorted(counts.tolist())==[1,2]
    averages=HardSimpleVFE(5)(voxels,counts)
    torch.testing.assert_close(averages[:,3].sort().values,torch.tensor([15.,30.],device='cuda'))
    features=torch.randn(4,128,device='cuda')
    indices=torch.tensor([[0,1,1,1],[0,2,1,1],[0,1,2,1],[0,1,1,2]],device='cuda',dtype=torch.int32)
    sparse=spconv.SparseConvTensor(features,indices,[4,4,4],1)
    layer=spconv.SubMConv3d(128,128,5,padding=2,bias=False).cuda().eval()
    result=layer(sparse).features
    assert result.shape==(4,128) and torch.isfinite(result).all()
    aggregate=LocalAggregator(3,4,4,4,[0,0,0],1).cuda()
    xyz=torch.tensor([[[1.5,1.5,1.5]]],device='cuda')
    logits=aggregate(xyz,xyz,torch.ones(1,1,device='cuda'),torch.ones(1,1,18,device='cuda'),
                     torch.ones(1,1,3,device='cuda'),torch.eye(3,device='cuda')[None,None])
    torch.testing.assert_close(logits,torch.ones_like(logits),atol=1e-5,rtol=1e-5)
    assert logits.numel()==18
    lifter=GaussianLifterLiDAR(num_anchor=8,embed_dims=4,semantics=True,semantic_dim=17).cuda().eval()
    out=lifter(ms_img_feats=[torch.zeros(1,6,4,2,2,device='cuda')],voxel_lidar_feats=points,
               coors_batch=torch.zeros(3,4,dtype=torch.int32,device='cuda'))
    anchors=safe_sigmoid(out['representation'][0,:,:3])
    targets=point_cloud_map(points)
    assert (torch.cdist(targets,anchors).min(1).values<1e-4).all()
torch.cuda.synchronize()
report={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'modulated_deform_conv_zero_offset_parity':True,'voxelization_counts':True,
        'hard_voxel_feature_mean':True,
        'spconv_gpu_forward':True,'gaussian_aggregate_gpu_forward':True,
        'sparse_lidar_anchor_assignment':True,
        'dfa3d_symbols_present':all(hasattr(dfa_ext,name) for name in ['ms_depth_score_sample_forward','wms_deform_attn_forward'])}
(ROOT/'reports/cuda_smoke.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
