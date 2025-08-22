import torch

from .graph import knn


def outlier_removal(points, radii, nb_points=4):
    idxs, dists, _ = knn(points, points, K=nb_points, r=torch.max(radii).item())

    keep = (dists < radii) & (idxs != -1)

    return keep.sum(1) == nb_points
