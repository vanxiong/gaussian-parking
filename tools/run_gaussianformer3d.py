"""Frozen GaussianFormer3D inference on one calibrated Isaac Sim capture."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from mmengine import Config
from mmengine.registry import MODELS
from prepare_capture import prepare, ORDER

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'GaussianFormer3D'))


def build_model(progress=None):
    progress = progress or (lambda stage: None)
    progress('importing model components')
    import model  # registers the model and compatibility builders
    progress('reading model configuration')
    config = Config.fromfile(str(ROOT / 'GaussianFormer3D/config/nuscenes_surroundocc_gs25600.py'))
    # The complete trained checkpoint supplies all backbone weights; no initializer download.
    config.model.img_backbone.pretrained = None
    config.model.img_backbone.init_cfg = None
    progress('constructing network')
    network = MODELS.build(config.model)
    checkpoint = ROOT / 'checkpoints/surroundocc_release.pth'
    progress('reading pretrained checkpoint')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = saved.get('state_dict', saved)
    state = {name.removeprefix('module.'): value for name, value in state.items()}
    progress('matching checkpoint tensors')
    network.load_state_dict(state, strict=True)
    progress('initializing CUDA weights')
    network.requires_grad_(False).eval().cuda()
    return network, config, len(state)


def build_queries(config):
    low = torch.tensor(config.pc_range[:3], device='cuda')
    high = torch.tensor(config.pc_range[3:], device='cuda')
    shape = (200, 200, 16)
    axes = [low[i] + (torch.arange(n, device='cuda') + .5) * (high[i]-low[i]) / n
            for i, n in enumerate(shape)]
    xyz = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)[None]
    return xyz


def infer_capture(network, config, capture, queries=None):
    arrays = prepare(capture)
    raw, confidence, stats = infer_arrays(network, config, arrays, queries)
    metadata = json.loads((capture / 'capture.json').read_text())
    reference = np.asarray(metadata['world_from_ego']) @ np.asarray(metadata['lidar']['ego_from_lidar'])
    return raw, confidence, reference, metadata, stats


def infer_arrays(network, config, arrays, queries=None):
    """Frozen inference on memory-backed inputs, independent of file transport."""
    images = torch.from_numpy(arrays['img'])[None].cuda()
    points = [torch.from_numpy(arrays['points']).cuda()]
    depth = torch.from_numpy(arrays['dpt'])[None].cuda()
    xyz = build_queries(config) if queries is None else queries
    metas = {'projection_mat': torch.from_numpy(arrays['projection_mat'])[None].cuda(),
             'image_wh': torch.from_numpy(arrays['image_wh'])[None].cuda(), 'occ_xyz': xyz}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = network(imgs=images, points=points, dpt=depth, metas=metas)
        logits = output['pred_occ'][-1]
        if not torch.isfinite(logits).all():
            raise RuntimeError('Nonfinite occupancy output')
        confidence, labels = logits.softmax(1).max(1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    raw = labels.reshape(200,200,16).cpu().numpy().astype(np.uint8)
    confidence = confidence.reshape(200,200,16).cpu().numpy().astype(np.float32)
    stats = {'inference_seconds':elapsed,'peak_cuda_allocated_GB':torch.cuda.max_memory_allocated()/1e9,
             'lidar_intensity_min_max':[float(arrays['points'][:,3].min()),float(arrays['points'][:,3].max())]}
    return raw,confidence,stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path, nargs='?')
    parser.add_argument('--build-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    started = time.perf_counter()
    network, config, tensors = build_model()
    print('Strict checkpoint match:', tensors, 'tensors', flush=True)
    (ROOT / 'reports/model_build.json').write_text(json.dumps(
        {'strict_tensor_count': tensors, 'frozen': not any(p.requires_grad for p in network.parameters()),
         'device': torch.cuda.get_device_name(), 'seconds': time.perf_counter()-started}, indent=2))
    if args.build_only:
        return
    if args.capture is None:
        parser.error('capture is required unless --build-only is used')
    raw,confidence,world_from_lidar,metadata,stats=infer_capture(network,config,args.capture)
    destination = args.capture / 'gaussianformer3d_prediction'
    destination.mkdir(exist_ok=True)
    np.save(destination / 'raw_labels.npy', raw)
    np.save(destination / 'confidence.npy', confidence)
    checkpoint = ROOT / 'checkpoints/surroundocc_release.pth'
    report = {'model': 'GaussianFormer3D', 'checkpoint_sha256': hashlib.file_digest(checkpoint.open('rb'), 'sha256').hexdigest(),
              'strict_tensor_count': tensors, 'frozen': True, 'shape': list(raw.shape),
              'range': list(config.pc_range), 'voxel_size': [.5,.5,.5],
              'world_from_lidar': world_from_lidar.tolist(), 'camera_order': ORDER,
              **stats,
              'timing_scope': 'first forward including softmax; excludes loading, capture, input preparation and output saving',
              'warmup_forwards': 0,
              'class_counts': np.bincount(raw.ravel(), minlength=18).tolist(),
              'raw_label_convention': '0 noise/ignore, 1..16 semantic classes, 17 empty',
              'ground_truth_labels_used': False, 'lidar': metadata['lidar'],
              'input_validation_passed': True,
              'limitations': 'One synthetic static capture, simplified LiDAR intensity and cameras; accuracy and real-time performance not established.'}
    (destination / 'prediction.json').write_text(json.dumps(report, indent=2))
    (ROOT / 'reports/latest_prediction.json').write_text(json.dumps(
        {'capture': str(args.capture.relative_to(ROOT)), 'prediction': str(destination.relative_to(ROOT))}, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
