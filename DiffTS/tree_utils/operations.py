import numpy as np

from typing import List

from .tube import Tube


def sample_tubes(tubes: List[Tube], sample_rate):

    pts, radius = [], []

    for i, tube in enumerate(tubes):

        start = tube.a
        end = tube.b

        start_rad = tube.r1
        end_rad = tube.r2

        v = end - start
        length = np.linalg.norm(v)
        direction = v / length
        num_pts = int(np.round(length / sample_rate))

        if num_pts > 0:

            # np.arange(0, float(length),  step=float(sample_rate)).reshape(-1,1)
            spaced_points = np.linspace(0, length, num_pts).reshape(-1, 1)

            lin_radius = np.linspace(
                start_rad, end_rad, spaced_points.shape[0], dtype=float)

            pts.append(start + direction * spaced_points)
            radius.append(lin_radius)

    return np.concatenate(pts, axis=0), np.concatenate(radius, axis=0)


def sample_o3d_lineset(ls, sample_rate):
    edges = np.asarray(ls.lines)
    xyz = np.asarray(ls.points)

    pts, radius = [], []

    for i, edge in enumerate(edges):

        start = xyz[edge[0]]
        end = xyz[edge[1]]

        v = end - start
        length = np.linalg.norm(v)
        direction = v / length
        num_pts = int(np.round(length / sample_rate))

        if num_pts > 0:

            # np.arange(0, float(length),  step=float(sample_rate)).reshape(-1,1)
            spaced_points = np.linspace(0, length, num_pts).reshape(-1, 1)
            pts.append(start + direction * spaced_points)

    return np.concatenate(pts, axis=0)

def sample_o3d_lineset_with_ids(ls, sample_rate):
    sampled_points = []
    sampled_ids = []
    for branch, branch_id in ls:
        edges = np.asarray(branch.lines)
        xyz = np.asarray(branch.points)

        pts, radius = [], []

        for i, edge in enumerate(edges):

            start = xyz[edge[0]]
            end = xyz[edge[1]]

            v = end - start
            length = np.linalg.norm(v)
            direction = v / length
            num_pts = int(np.round(length / sample_rate))

            if num_pts > 0:

                # np.arange(0, float(length),  step=float(sample_rate)).reshape(-1,1)
                spaced_points = np.linspace(0, length, num_pts).reshape(-1, 1)
                pts.append(start + direction * spaced_points)
        sampled_points.append(np.concatenate(pts, axis=0))
        sampled_ids.append(np.full(np.concatenate(pts, axis=0).shape[0], branch_id))
    return np.concatenate(sampled_points, axis=0), np.concatenate(sampled_ids, axis=0)

def sample_graph(ls, sample_rate):

    edges = np.asarray(ls.lines)
    xyz = np.asarray(ls.points)

    pts, radius = [], []

    for i, edge in enumerate(edges):

        start = xyz[edge[0]]
        end = xyz[edge[1]]

        v = end - start
        length = np.linalg.norm(v)
        direction = v / length
        num_pts = int(np.round(length / sample_rate))

        if num_pts > 0:

            # np.arange(0, float(length),  step=float(sample_rate)).reshape(-1,1)
            spaced_points = np.linspace(0, length, num_pts).reshape(-1, 1)
            pts.append(start + direction * spaced_points)

    return np.concatenate(pts, axis=0)