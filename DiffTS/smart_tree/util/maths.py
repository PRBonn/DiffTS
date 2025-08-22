from typing import List

import numpy as np
import torch
import torch.nn.functional as F


def np_normalized(a: np.array, axis=-1, order=2) -> np.array:
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1e-13
    return a / np.expand_dims(l2, axis), l2


def torch_normalized(v):
    return F.normalize(v), v.pow(2).sum(1).sqrt().unsqueeze(1)


def euler_angles_to_rotation(xyz: List) -> torch.tensor:
    x, y, z = xyz

    R_X = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, torch.cos(x), -torch.sin(x)],
            [0.0, torch.sin(x), torch.cos(x)],
        ]
    )

    R_Y = torch.tensor(
        [
            [torch.cos(y), 0.0, torch.sin(y)],
            [0.0, 1.0, 0.0],
            [-torch.sin(y), 0.0, torch.cos(y)],
        ]
    )

    R_Z = torch.tensor(
        [
            [torch.cos(z), -torch.sin(z), 0.0],
            [torch.sin(z), torch.cos(z), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    return torch.mm(R_Z, torch.mm(R_Y, R_X))


def rotation_matrix_from_vectors_np(vec1: np.array, vec2: np.array) -> np.array:
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
        vec2 / np.linalg.norm(vec2)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def rotation_matrix_from_vectors_torch(vec1, vec2):
    a, b = F.normalize(vec1, dim=0), F.normalize(vec2, dim=0)
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    s = torch.linalg.norm(v)

    kmat = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = torch.eye(3) + kmat + kmat.matmul(kmat) * ((1 - c) / (s**2))

    return rotation_matrix


def make_transformation_matrix(rotation, translation):
    return torch.vstack(
        (torch.hstack((rotation, translation)), torch.tensor([0.0, 0.0, 0.0, 1.0]))
    )


def bb_filter(
    points,
    min_x=-np.inf,
    max_x=np.inf,
    min_y=-np.inf,
    max_y=np.inf,
    min_z=-np.inf,
    max_z=np.inf,
):
    bound_x = np.logical_and(points[:, 0] >= min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] >= min_y, points[:, 1] < max_y)
    bound_z = np.logical_and(points[:, 2] >= min_z, points[:, 2] < max_z)

    bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    return bb_filter


"""
def cube_filter(points, center, cube_size):
  
  min_x = center[0] - cube_size / 2
  max_x = center[0] + cube_size / 2
  min_y = center[1] - cube_size / 2
  max_y = center[1] + cube_size / 2
  min_z = center[2] - cube_size / 2
  max_z = center[2] + cube_size / 2

  return bb_filter(points, min_x, max_x, min_y, max_y, min_z, max_z)
"""


def np_bb_filter(
    points,
    min_x=-np.inf,
    max_x=np.inf,
    min_y=-np.inf,
    max_y=np.inf,
    min_z=-np.inf,
    max_z=np.inf,
):
    bound_x = np.logical_and(points[:, 0] >= min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] >= min_y, points[:, 1] < max_y)
    bound_z = np.logical_and(points[:, 2] >= min_z, points[:, 2] < max_z)

    bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    return bb_filter


def torch_bb_filter(points, min_x, max_x, min_y, max_y, min_z, max_z):
    bound_x = torch.logical_and(points[:, 0] >= min_x, points[:, 0] < max_x)
    bound_y = torch.logical_and(points[:, 1] >= min_y, points[:, 1] < max_y)
    bound_z = torch.logical_and(points[:, 2] >= min_z, points[:, 2] < max_z)

    bb_filter = torch.logical_and(torch.logical_and(bound_x, bound_y), bound_z)

    return bb_filter


def cube_filter(points, center, cube_size):
    min = center - (cube_size / 2)
    max = center + (cube_size / 2)

    if type(center) == np.array:
        return np_bb_filter(points, min[0], max[0], min[1], max[1], min[2], max[2])

    max = max.to(points.device)
    min = min.to(points.device)

    return torch_bb_filter(points, min[0], max[0], min[1], max[1], min[2], max[2])


def vertex_dirs(points):
    d = points[1:] - points[:-1]
    d = d / np.linalg.norm(d)

    smooth = (d[1:] + d[:-1]) * 0.5
    dirs = np.concatenate([np.array(d[0:1]), smooth, np.array(d[-2:-1])])

    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def random_unit(dtype=np.float32):
    x = np.random.randn(3).astype(dtype)
    return x / np.linalg.norm(x)


def make_tangent(d, n):
    t = np.cross(d, n)
    t /= np.linalg.norm(t, axis=-1, keepdims=True)
    return np.cross(t, d)


def gen_tangents(dirs, t):
    tangents = []

    for dir in dirs:
        t = make_tangent(dir, t)
        tangents.append(t)

    return np.stack(tangents)
