from itertools import repeat
from typing import List, Tuple, Union

import numpy as np
import spconv.pytorch as spconv
import torch


def sparse_from_batch(features, coordinates, device):
    batch_size = features.shape[0]

    features = features.to(device)
    coordinates = coordinates.to(device)

    values, _ = torch.max(coordinates, 0)  # BXYZ -> XYZ (Biggest Spatial Size)

    return spconv.SparseConvTensor(
        features, coordinates.int(), values[1:], batch_size=batch_size
    )


def split_sparse_list(indices, features):
    cloud_ids = indices[:, 0]

    num_clouds = cloud_ids.max() + 1
    return [
        (indices[cloud_ids == i], features[cloud_ids == i]) for i in range(num_clouds)
    ]


def split_sparse(sparse_tensor):
    cloud_ids = sparse_tensor.indices[:, 0]
    num_clouds = cloud_ids.max() + 1
    return [
        (sparse_tensor.indices[cloud_ids == i], sparse_tensor.features[cloud_ids == i])
        for i in range(num_clouds)
    ]


def batch_collate(batch):
    """Custom Batch Collate Function for Sparse Data..."""

    batch_feats, batch_coords, batch_mask, fn = zip(*batch)

    for i, coords in enumerate(batch_coords):
        coords[:, 0] = torch.tensor([i], dtype=torch.float32)

    if isinstance(batch_feats[0], tuple):
        input_feats, target_feats = tuple(zip(*batch_feats))

        input_feats, target_feats, coords, mask = [
            torch.cat(x) for x in [input_feats, target_feats, batch_coords, batch_mask]
        ]

        return [(input_feats, target_feats), coords, mask, fn]

    feats, coords, mask = [
        torch.cat(x) for x in [batch_feats, batch_coords, batch_mask]
    ]

    return [feats, coords, mask, fn]


def ravel_hash(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, x.shape

    x = x - np.min(x, axis=0)
    x = x.astype(np.uint64, copy=False)
    xmax = np.max(x, axis=0).astype(np.uint64) + 1

    h = np.zeros(x.shape[0], dtype=np.uint64)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h


def sparse_quantize(
    coords,
    voxel_size: Union[float, Tuple[float, ...]] = 1,
    *,
    return_index: bool = False,
    return_inverse: bool = False,
) -> List[np.ndarray]:
    if isinstance(voxel_size, (float, int)):
        voxel_size = tuple(repeat(voxel_size, 3))
    assert isinstance(voxel_size, tuple) and len(voxel_size) == 3

    voxel_size = np.array(voxel_size)
    coords = np.floor(coords / voxel_size).astype(np.int32)

    _, indices, inverse_indices = np.unique(
        ravel_hash(coords), return_index=True, return_inverse=True
    )
    coords = coords[indices]

    outputs = [coords]
    if return_index:
        outputs += [indices]
    if return_inverse:
        outputs += [inverse_indices]
    return outputs[0] if len(outputs) == 1 else outputs
