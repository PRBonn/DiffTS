import numpy as np
import open3d as o3d
import torch

# from o3d_abstractions.viewimport o3d_view_geometries
from smart_tree.util.maths import (gen_tangents, random_unit,
                                   vertex_dirs)


def o3d_mesh(verts, tris):
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris)
    ).compute_triangle_normals()


def o3d_merge_meshes(meshes, colourize=False):
    if colourize:
        for m in meshes:
            m.paint_uniform_color(np.random.rand(3))

    mesh = meshes[0]
    for m in meshes[1:]:
        mesh += m
    return mesh


def o3d_merge_linesets(line_sets, colour=[0, 0, 0]):
    sizes = [np.asarray(ls.points).shape[0] for ls in line_sets]
    offsets = np.cumsum([0] + sizes)

    points = np.concatenate([ls.points for ls in line_sets])
    idxs = np.concatenate([ls.lines + offset for ls, offset in zip(line_sets, offsets)])

    return o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(points), o3d.utility.Vector2iVector(idxs)
    ).paint_uniform_color(colour)


def points_to_edge_idx(points):
    idx = torch.arange(points.shape[0] - 1)
    return torch.column_stack((idx, idx + 1))


def o3d_sphere(xyz, radius, colour=(1, 0, 0)):
    return (
        o3d.geometry.TriangleMesh.create_sphere(radius)
        .translate(np.asarray(xyz))
        .paint_uniform_color(colour)
    )


def o3d_spheres(xyzs, radii, colour=None, colours=None):
    spheres = [o3d_sphere(xyz, r) for xyz, r in zip(xyzs, radii)]
    return paint_o3d_geoms(spheres, colour, colours)


def o3d_line_set(vertices, edges, colour=None):
    line_set = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector2iVector(edges)
    )
    if colour is not None:
        return line_set.paint_uniform_color(colour)
    return line_set


def o3d_line_sets(vertices, edges):
    return [o3d_line_set(v, e) for v, e in zip(vertices, edges)]


def o3d_path(vertices, colour=None):
    edge_idx = points_to_edge_idx(vertices)
    if colour is not None:
        return o3d_line_set(vertices, edge_idx).paint_uniform_color(colour)
    return o3d_line_set(vertices, edge_idx)


def o3d_paths(vertices):
    return [o3d_path(v) for v in vertices]


def o3d_merge_clouds(points_clds):
    points = np.concatenate([np.asarray(pcd.points) for pcd in points_clds])
    colors = np.concatenate([np.asarray(pcd.colors) for pcd in points_clds])

    return o3d_cloud(points, colours=colors)


def o3d_cloud(points, colour=None, colours=None, normals=None):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals)
    if colour is not None:
        return cloud.paint_uniform_color(colour)
    elif colours is not None:
        cloud.colors = o3d.utility.Vector3dVector(colours)
        return cloud

    return cloud.paint_uniform_color([1, 0, 0])


def class_label_o3d_cloud(cloud, class_labels, cmap=[]):
    cloud.colors = o3d.utility.Vector3dVector(np.asarray(cmap)[class_labels])
    return cloud


def o3d_clouds(batch_points, colour=None, colours=None, p_colours=None):
    # KN#
    # colour -> paint all the same colour
    # colours -> colour for each point cloud
    # p_colours -> points for each point in each cloud
    if colour is not None:
        return [o3d_cloud(p, colour) for p in zip(batch_points)]

    if colours is not None:
        return [o3d_cloud(p, c) for p, c in zip(batch_points, colours)]

    if p_colours is not None:
        return [o3d_cloud(p, colours=c) for p, c in zip(batch_points, p_colours)]

    return [o3d_cloud(p, n) for p, n in zip(batch_points)]


def o3d_voxel_grid(
    width: float,
    depth: float,
    height: float,
    voxel_size: float,
    origin=np.asarray([0, 0, 0]),
    colour=np.asarray([1, 1, 0]),
):
    return o3d.geometry.VoxelGrid.create_dense(
        origin, colour, voxel_size, width, depth, height
    )


def o3d_cylinder(radius, length, colour=(1, 0, 0)):
    return o3d.geometry.TriangleMesh.create_cylinder(
        radius, length
    ).paint_uniform_color(colour)


def o3d_cylinders(radii, length, colour=None, colours=None):
    cylinders = [o3d_cylinder(r, l, colour) for r, l in zip(radii, length)]
    return paint_o3d_geoms(cylinders, colour, colours)


def paint_o3d_geoms(geometries, colour=None, colours=None):
    if colours is not None:
        return [
            geom.paint_uniform_color(colours[i]) for i, geom in enumerate(geometries)
        ]
    elif colour is not None:
        return [geom.paint_uniform_color(colour) for geom in geometries]
    return geometries


def unit_circle(n):
    a = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    return np.stack([np.sin(a), np.cos(a)], axis=1)


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


def o3d_lines_between_clouds(cld1, cld2):
    pts1 = np.asarray(cld1.points)
    pts2 = np.asarray(cld2.points)

    interweaved = np.hstack((pts1, pts2)).reshape(-1, 3)
    return o3d_line_set(
        interweaved, np.arange(0, min(pts1.shape[0], pts2.shape[0]) * 2).reshape(-1, 2)
    )


def o3d_tube_mesh(points, radii, colour=(1, 0, 0), n=10):
    points = tube_vertices(points, radii, n)

    n, m, _ = points.shape
    indexes = cylinder_triangles(m, n)

    mesh = o3d_mesh(points.reshape(-1, 3), indexes)
    mesh.compute_vertex_normals()

    return mesh.paint_uniform_color(colour)


def sample_o3d_lineset(lineset, sample_rate):
    edges = np.asarray(lineset.lines)
    xyz = np.asarray(lineset.points)

    pts, radius = [], []

    for i, edge in enumerate(edges):
        start = xyz[edge[0]]
        end = xyz[edge[1]]

        v = end - start
        length = np.linalg.norm(v)
        direction = v / length
        num_points = np.ceil(length / sample_rate)

        if int(num_points) > 0.0:
            spaced_points = np.arange(
                0, float(length), step=float(length / num_points)
            ).reshape(-1, 1)
            pts.append(start + direction * spaced_points)

    return np.concatenate(pts, axis=0)
