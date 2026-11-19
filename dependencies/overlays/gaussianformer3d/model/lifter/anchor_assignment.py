"""Repeatable evaluation assignment without modifying learned anchor parameters."""
from functools import lru_cache

import torch


@lru_cache(maxsize=16)
def _permutation(size):
    # A private CPU generator keeps assignment independent of global RNG state,
    # warm-up forwards, and CUDA RNG implementations. Never mutate this tensor.
    generator = torch.Generator(device='cpu').manual_seed(42)
    return torch.randperm(size, generator=generator)


def evaluation_assignment(coordinates, num_anchors):
    """Return paired anchor/voxel indices, independent of voxel output order.

    Coordinates are unique integer (z, y, x) voxel coordinates for one sample.
    Preserve the upstream sparse/dense sampling policy and learned fallback
    anchors, while choosing the same assignment for the same occupied voxels.
    """
    count = coordinates.shape[0]
    if not count:
        empty = torch.empty(0, dtype=torch.long, device=coordinates.device)
        return empty, empty
    xyz = coordinates.to(torch.int64)
    extent = xyz.amax(dim=0) + 1
    key = (xyz[:, 0] * extent[1] + xyz[:, 1]) * extent[2] + xyz[:, 2]
    canonical = torch.argsort(key)
    if num_anchors > count:
        destinations = _permutation(num_anchors)[:count].to(coordinates.device)
        return destinations, canonical
    selected = _permutation(count)[:num_anchors].to(coordinates.device)
    destinations = torch.arange(num_anchors, device=coordinates.device)
    return destinations, canonical[selected]
