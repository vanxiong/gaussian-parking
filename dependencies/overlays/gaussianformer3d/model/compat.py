"""Unify upstream's mixed old/new OpenMMLab registries without changing weights."""
from types import SimpleNamespace
import torch
from torch import nn
from mmengine.registry import MODELS
from mmdet.models.backbones import ResNet
from mmdet.models.necks import FPN

SEGMENTORS = HEADS = MODELS
build_backbone = build_head = MODELS.build
builder = SimpleNamespace(build_backbone=MODELS.build, build_neck=MODELS.build,
                          build_head=MODELS.build)
for cls in (ResNet, FPN):
    MODELS.register_module(module=cls, force=True)


@MODELS.register_module()
class HardSimpleVFE(nn.Module):
    """Parameter-free mmdet3d HardSimpleVFE: mean valid features per voxel."""
    def __init__(self, num_features=4):
        super().__init__()
        self.num_features = num_features

    @torch.no_grad()
    def forward(self, features, num_points, coors=None):
        return features[:, :, :self.num_features].sum(1).contiguous() / num_points.to(features.dtype).view(-1, 1)
