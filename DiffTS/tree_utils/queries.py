import numpy as np
import torch

from typing import List

from .tube import Tube, CollatedTube, collate_tubes

from pykeops.torch import LazyTensor

""" 
For the following :
  N : number of pts
  M : number of tubes
"""


# N x 3, M x 2
def points_to_collated_tube_projections(pts: np.array, collated_tube: CollatedTube, eps=1e-12):

    ab = collated_tube.b - collated_tube.a  # M x 3

    ap = pts[:, np.newaxis] - collated_tube.a[np.newaxis, ...]  # N x M x 3

    t = np.clip(np.einsum('nmd,md->nm', ap, ab) /
                (np.einsum('md,md->m', ab, ab) + eps), 0.0, 1.0)  # N x M
    proj = collated_tube.a[np.newaxis, ...] + \
        np.einsum('nm,md->nmd', t, ab)  # N x M x 3
    return proj, t


def projection_to_distance_matrix(projections, pts):  # N x M x 3
    # N x M
    return np.sqrt(np.sum(np.square(projections - pts[:, np.newaxis, :]), 2))


def pts_to_nearest_tube(pts: np.array, tubes: List[Tube]):
    """ Vectors from pt to the nearest tube """

    collated_tube = collate_tubes(tubes)
    projections, t = points_to_collated_tube_projections(
        pts, collated_tube)  # N x M x 3

    r = (1 - t) * collated_tube.r1 + t * collated_tube.r2

    distances = projection_to_distance_matrix(projections, pts)   # N x M

    distances = (distances - r)
    idx = np.argmin(distances, 1)  # N

    # assert idx.shape[0] == pts.shape[0]

    # vector, idx , radius
    return projections[np.arange(pts.shape[0]), idx] - pts, idx, r[np.arange(pts.shape[0]), idx]


def pts_on_nearest_tube(pts: np.array, tubes: List[Tube]):

    vectors, index, radius = pts_to_nearest_tube(pts, tubes)
    return pts + vectors, radius


# def knn(src, dest, K=50, r=1.0, grid=None):
#     src_lengths = src.new_tensor([src.shape[0]], dtype=torch.long)
#     dest_lengths = src.new_tensor([dest.shape[0]], dtype=torch.long)
#     dists, idxs, grid, _ = frnn.frnn_grid_points(
#         src.unsqueeze(0), dest.unsqueeze(0),
#         src_lengths, dest_lengths,
#         K, r,return_nn=False, return_sorted=True)
#     return idxs.squeeze(0), dists.sqrt().squeeze(0), grid


# def nn_frnn(src, dest, r=1.0, grid=None):
#   idx, dist, grid = knn(src, dest, K=1, r=r, grid=grid)
#   idx, dist = idx.squeeze(1), dist.squeeze(1)
#   return idx, dist, grid


def distance_matrix_keops(pts1, pts2, device=torch.device("cuda")):

    pts1 = pts1.clone().to(device).float()
    pts2 = pts2.clone().to(device).float()

    x_i = LazyTensor(pts1.reshape(-1, 1, 3).float())
    y_j = LazyTensor(pts2.view(1, -1, 3).float())

    return (x_i - y_j).square().sum(dim=2).sqrt()


def nn_keops(pts1, pts2):

    D_ij = distance_matrix_keops(pts1, pts2)

    return D_ij.min(1), D_ij.argmin(1).flatten()  # distance, idx
