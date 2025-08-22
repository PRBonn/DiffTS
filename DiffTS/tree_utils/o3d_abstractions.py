from dataclasses import dataclass, asdict
from typing import Sequence

import open3d as o3d

import numpy as np

from .math import unit_circle, vertex_dirs, gen_tangents, random_unit


def o3d_cloud(points, colour=None, colours=None, normals=None):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals)
    if colour is not None:
        return cloud.paint_uniform_color(colour)
    elif colours is not None:
        cloud.colors = o3d.utility.Vector3dVector(colours)

    return cloud


def o3d_merge_linesets(line_sets, colour=(0, 0, 0)):
    sizes = [np.asarray(ls.points).shape[0] for ls in line_sets]
    offsets = np.cumsum([0] + sizes)

    points = np.concatenate([ls.points for ls in line_sets])
    idxs = np.concatenate([ls.lines + offset for ls, offset in zip(line_sets, offsets)])

    return o3d_line_set(points, idxs).paint_uniform_color(colour)

def o3d_merge_linesets_with_ids(line_sets, colour=(0, 0, 0)):
    import ipdb;ipdb.set_trace()  # fmt: skip
    line_sets, ids = zip(*line_sets)
    sizes = [np.asarray(ls.points).shape[0] for ls in line_sets]
    offsets = np.cumsum([0] + sizes)

    points = np.concatenate([ls.points for ls in line_sets])
    idxs = np.concatenate([ls.lines + offset for ls, offset in zip(line_sets, offsets)])
    branch_ids = np.concatenate([np.ones(np.asarray(ls.points).shape[0]) * id for ls, id in zip(line_sets, ids)])

    return o3d_line_set(points, idxs).paint_uniform_color(colour)


def o3d_line_set(vertices, edges, colour=None):
    ls = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector2iVector(edges)
    )
    if colour is not None:
        return ls.paint_uniform_color(colour)
    return ls


def o3d_path(vertices, colour=None):
    idx = np.arange(vertices.shape[0] - 1)
    edge_idx = np.column_stack((idx, idx + 1))
    if colour is not None:
        return o3d_line_set(vertices, edge_idx, colour)
    return o3d_line_set(vertices, edge_idx)


def o3d_merge_meshes(meshes):
    sizes = [np.asarray(mesh.vertices).shape[0] for mesh in meshes]
    offsets = np.cumsum([0] + sizes)

    part_indexes = np.repeat(np.arange(0, len(meshes)), sizes)

    triangles = np.concatenate(
        [mesh.triangles + offset for offset, mesh in zip(offsets, meshes)]
    )
    vertices = np.concatenate([mesh.vertices for mesh in meshes])

    mesh = o3d_mesh(vertices, triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.concatenate([np.asarray(mesh.vertex_colors) for mesh in meshes])
    )
    return mesh


def o3d_mesh(verts, tris):
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris)
    ).compute_triangle_normals()


def o3d_lines_between_clouds(cld1, cld2):
    pts1 = np.asarray(cld1.points)
    pts2 = np.asarray(cld2.points)

    interweaved = np.hstack((pts1, pts2)).reshape(-1, 3)
    return o3d_line_set(
        interweaved, np.arange(0, min(pts1.shape[0], pts2.shape[0]) * 2).reshape(-1, 2)
    )


def cylinder_triangles(m, n):
    tri1 = np.array([0, 1, 2])
    tri2 = np.array([2, 3, 0])

    v0 = np.arange(m)
    v1 = (v0 + 1) % m
    v2 = v1 + m
    v3 = v0 + m

    edges = np.stack([v0, v1, v2, v3], axis=1)

    segments = np.arange(n - 1) * m
    edges = edges.reshape(1, *edges.shape) + segments.reshape(n - 1, 1, 1)

    edges = edges.reshape(-1, 4)
    return np.concatenate([edges[:, tri1], edges[:, tri2]])


def tube_vertices(points, radii, n=10):
    circle = unit_circle(n).astype(np.float32)

    dirs = vertex_dirs(points)
    t = gen_tangents(dirs, random_unit())

    b = np.stack([t, np.cross(t, dirs)], axis=1)
    b = b * radii.reshape(-1, 1, 1)

    return np.einsum("bdx,md->bmx", b, circle) + points.reshape(points.shape[0], 1, 3)


def o3d_tube_mesh(points, radii, colour=(1, 0, 0), n=10):
    points = tube_vertices(points, radii, n)

    n, m, _ = points.shape
    indexes = cylinder_triangles(m, n)

    mesh = o3d_mesh(points.reshape(-1, 3), indexes)
    mesh.compute_vertex_normals()

    return mesh.paint_uniform_color(colour)


def o3d_load_lineset(path, colour=[0, 0, 0]):
    return o3d.io.read_line_set(path).paint_uniform_color(colour)


@dataclass
class ViewerItem:
    name: str
    geometry: o3d.geometry.Geometry
    is_visible: bool = True


def o3d_viewer(items: Sequence[ViewerItem], line_width=1):
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"

    line_mat = o3d.visualization.rendering.MaterialRecord()
    line_mat.shader = "unlitLine"
    line_mat.line_width = line_width

    def material(item):
        return line_mat if isinstance(item.geometry, o3d.geometry.LineSet) else mat

    geometries = [dict(**asdict(item), material=material(item)) for item in items]
    o3d.visualization.draw(geometries, line_width=line_width)
