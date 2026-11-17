"""Apply idempotent, weight-preserving compatibility fixes to local sources."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
model = ROOT / 'GaussianFormer3D/model'
compatibility = (ROOT/'tools/compat_openmmlab.py').read_text()
if not (model/'compat.py').exists() or (model/'compat.py').read_text() != compatibility:
    (model/'compat.py').write_text(compatibility)
for path in model.rglob('*.py'):
    if path.name == 'compat.py':
        continue
    original = text = path.read_text()
    text = text.replace('from mmseg.models import ', 'from model.compat import ')
    text = text.replace('from mmseg.registry import MODELS', 'from mmengine.registry import MODELS')
    text = text.replace('from mmdet3d.registry import MODELS', 'from mmengine.registry import MODELS')
    text = text.replace('from mmseg.models.backbones import *', 'from model.compat import ResNet')
    text = text.replace('from mmseg.models.necks import *', 'from model.compat import FPN')
    text = text.replace('from turtle import down, forward', '# Removed unused turtle import (headless worker has no Tk).')
    if path.name == 'spconv3d_module.py':
        text = re.sub(r'spatial_shape = indices.max\(0\)\[0\](?: \+ 1)*',
                      'spatial_shape = indices.max(0)[0] + 1',text)
    if path.name == 'gaussian_lifter.py':
        text = text.replace('anchor[batch_idx][idx][:, :3] = voxel_map_coords_logits',
                            'anchor[batch_idx, idx, :3] = voxel_map_coords_logits')
        text = text.replace('anchor[batch_idx][idx][:, 10] = voxel_map_intensity_logits',
                            'anchor[batch_idx, idx, 10] = voxel_map_intensity_logits')
    if path.name == 'gaussian_head.py':
        text = text.replace("""        occ_label = metas['occ_label'].to(self.zero_tensor.device)
        occ_cam_mask = metas['occ_cam_mask'].to(self.zero_tensor.device)
        sampled_xyz, sampled_label = self._sampling(occ_xyz, occ_label, None)""",
        """        # Inference queries geometric locations, never ground-truth labels.
        if 'occ_label' in metas:
            occ_label = metas['occ_label'].to(self.zero_tensor.device)
            occ_cam_mask = metas.get('occ_cam_mask')
            if occ_cam_mask is not None:
                occ_cam_mask = occ_cam_mask.to(self.zero_tensor.device)
            sampled_xyz, sampled_label = self._sampling(occ_xyz, occ_label, None)
        elif not self.training:
            sampled_xyz = occ_xyz.flatten(1, 3)
            sampled_label = occ_cam_mask = None
        else:
            raise ValueError('Training requires occupancy labels')""")
    if text != original:
        path.write_text(text)
        print(path.relative_to(ROOT))
setup = ROOT / 'vendor/DFA3D/DFA3D/setup.py'
text = setup.read_text().replace('-std=c++14', '-std=c++17')
setup.write_text(text)
for tree in [ROOT / 'vendor/DFA3D/DFA3D/dfa3D/ops/csrc',
             model / 'encoder/gaussian_encoder/ops', model / 'head/localagg']:
    for path in tree.rglob('*'):
        if path.suffix in ['.cpp', '.cu', '.h', '.cuh']:
            original = path.read_text()
            text = original.replace('.data<', '.data_ptr<')
            if text != original:
                path.write_text(text)
