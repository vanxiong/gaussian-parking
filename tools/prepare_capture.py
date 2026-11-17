"""Prepare calibrated camera/LiDAR arrays; this does not run GaussianFormer3D."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

ORDER = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
         'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def rigid(value):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Expected finite 4x4 transform')
    if not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError('Invalid homogeneous transform')
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError('Expected a proper rigid rotation')
    return matrix


def project_depth(points_xyz, optical_from_lidar, intrinsic, height, width):
    optical = points_xyz @ optical_from_lidar[:3, :3].T + optical_from_lidar[:3, 3]
    optical = optical[optical[:, 2] > 0.1]
    pixels = optical @ intrinsic.T
    uv = np.floor(pixels[:, :2] / pixels[:, 2:3]).astype(np.int64)
    good = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv, optical = uv[good], optical[good]
    depth = np.full((height, width), np.inf, dtype=np.float32)
    np.minimum.at(depth, (uv[:, 1], uv[:, 0]), optical[:, 2])
    depth[~np.isfinite(depth)] = 0
    return depth


def prepare(capture):
    metadata = json.loads((capture / 'capture.json').read_text())
    lidar = metadata.get('lidar')
    if not lidar:
        raise ValueError('Capture has no LiDAR record. RGB-only legacy captures cannot drive this fusion model.')
    if lidar.get('fields') != ['x', 'y', 'z', 'intensity', 'time_lag_seconds']:
        raise ValueError('Explicit LiDAR fields x,y,z,intensity,time_lag_seconds are required')
    points_path = capture / lidar['points']
    points = (np.load(points_path, allow_pickle=False) if points_path.suffix == '.npy'
              else np.fromfile(points_path, dtype='<f4').reshape(-1, 5))
    rgb = {camera['name']: np.asarray(Image.open(capture / camera['image']).convert('RGB'))
           for camera in metadata['cameras']}
    return prepare_arrays(metadata, rgb, points)


def prepare_arrays(metadata, rgb, points):
    """Shared preprocessing for saved captures and native ROS sensor messages."""
    lidar = metadata['lidar']
    if lidar.get('fields') != ['x', 'y', 'z', 'intensity', 'time_lag_seconds']:
        raise ValueError('Explicit LiDAR fields are required')
    if points.ndim != 2 or points.shape[1] != 5 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Expected nonempty finite Nx5 points in the current LiDAR frame')
    if (points[:,3]<0).any() or (points[:,3]>255).any() or (points[:,4]<0).any():
        raise ValueError('LiDAR intensity must be 0..255 and sweep time lags must be nonnegative')
    ego_from_lidar = rigid(lidar['ego_from_lidar'])
    cameras = {camera['name']: camera for camera in metadata['cameras']}
    if len(cameras) != 6 or set(cameras) != set(ORDER):
        raise ValueError('Exactly six named surround cameras are required')
    images, projections, depths, sizes = [], [], [], []
    for name in ORDER:
        camera = cameras[name]
        image = np.asarray(rgb[name], dtype=np.float32)
        height, width = image.shape[:2]
        if camera['resolution'] != [width, height]:
            raise ValueError('Camera resolution and image disagree: ' + name)
        intrinsic = np.asarray(camera['intrinsic'], dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0 or not np.allclose(intrinsic[2], [0, 0, 1]):
            raise ValueError('Invalid pinhole calibration: ' + name)
        optical_from_lidar = np.linalg.inv(rigid(camera['ego_from_optical'])) @ ego_from_lidar
        projection = np.eye(4, dtype=np.float32)
        projection[:3] = intrinsic @ optical_from_lidar[:3]
        # Match upstream RGB normalization, then bottom/right zero padding.
        normalized = (image - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
        pad_h, pad_w = (-height) % 32, (-width) % 32
        images.append(np.pad(normalized.transpose(2, 0, 1), ((0, 0), (0, pad_h), (0, pad_w))))
        # Temporal sweep points are useful to the voxel encoder, but do not supply
        # stale returns as current depth guidance in this initial adapter.
        current = points[np.isclose(points[:, 4], 0), :3]
        depth = project_depth(current, optical_from_lidar, intrinsic, height, width)
        depths.append(np.pad(depth, ((0, pad_h), (0, pad_w)))[None])
        projections.append(projection)
        sizes.append([width + pad_w, height + pad_h])
    if len({image.shape for image in images}) != 1:
        raise ValueError('Camera images must have the same padded dimensions')
    return dict(img=np.stack(images).astype(np.float32), points=points.astype(np.float32),
                projection_mat=np.stack(projections), image_wh=np.asarray(sizes, dtype=np.float32),
                dpt=np.stack(depths).astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    arrays = prepare(args.capture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(json.dumps({'arrays': {name: list(value.shape) for name, value in arrays.items()},
                      'model_inference': False, 'status': 'Prepared inputs; runtime integration unverified'}, indent=2))


if __name__ == '__main__':
    main()
