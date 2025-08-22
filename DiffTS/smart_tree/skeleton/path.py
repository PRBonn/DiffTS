
import torch
from tqdm import tqdm

from ..data_types.branch import BranchSkeleton
from .graph import nn


def trace_route(preds, idx, termination_pts):
    path = []

    while idx >= 0 and idx not in termination_pts:
        path.append(idx)
        idx = preds[idx]

    return preds.new_tensor(path, dtype=torch.long).flip(0), idx


def select_path_points(
    points: torch.tensor, path_verts: torch.tensor, radii: torch.tensor
):
    """
    Finds points nearest to a path (specified by points with radii).
    points: (N, 3) 3d points of point cloud
    path_verts: (M, 3) 3d points of path
    radii: (M, 1) radii of path
    returns: (X, 2) index tuples of (path, point) for X points falling within the path, ordered by path index
    """

    point_path, dists, _ = nn(
        points,
        path_verts,
        r=radii.max().item(),
    )  # nearest path idx for each point
    valid = dists[point_path >= 0] < radii[point_path[point_path >= 0]].squeeze(
        1
    )  # where the path idx is less than the distance to the point

    on_path = point_path.new_zeros(point_path.shape, dtype=torch.bool)
    on_path[point_path >= 0] = valid  # points that are on the path.

    idx_point = on_path.nonzero().squeeze(1)
    idx_path = point_path[idx_point]

    order = torch.argsort(idx_path)
    return idx_point[order], idx_path[order]


def sample_tree(
    medial_pts,
    medial_radii,
    preds,
    distances,
    all_points,
    root_idx=0,
    visualize=False,
    pbar=None,
):
    """
    Medial Points: NN estimated medial points
    Medial Radii: NN estimated radii of points
    Preds: Predecessor of each medial point (on path to root node)
    Distance: Distance from root node to medial points
    Surface Points: The point the medial pts got projected from..
    """

    branch_id = 0

    branches = {}

    selection_mask = preds > 0
    distances[~selection_mask] = -1

    termination_pts = torch.tensor([], device=torch.device("cuda"))
    branch_ids = torch.full(
        (medial_pts.shape[0],),
        -1,
        device=torch.device("cuda"),
        dtype=int,
    )

    pbar = tqdm(
        total=distances.shape[0],
        leave=False,
        desc="Allocating Points",
    )

    while True:
        pbar.update(n=((distances < 0).sum().item() - pbar.n))
        pbar.refresh()
        
        farthest = distances.argmax().item()

        if distances[farthest] <= 0:
            break

        """ Traces the path of the futhrest point until it converges with allocated points """
        path_vertices_idx, termination_idx = trace_route(
            preds,
            farthest,
            termination_pts,
        )

        """ Gets the points around that path (and which path indexs they are close to) """
        idx_points, idx_path = select_path_points(
            medial_pts,
            medial_pts[path_vertices_idx],
            medial_radii[path_vertices_idx],
        )

        """ Mark this points as allocated and as termination points """
        distances[idx_points] = -1
        distances[path_vertices_idx] = -1
        termination_pts = torch.unique(
            torch.cat(
                (
                    termination_pts,
                    idx_points,
                    path_vertices_idx,
                )
            )
        )

        """ If the path has at least two points, save it as a branch """
        if len(path_vertices_idx) < 2:
            continue

        branches[branch_id] = BranchSkeleton(
            branch_id,
            xyz=medial_pts[path_vertices_idx].cpu(),
            radii=medial_radii[path_vertices_idx].cpu(),
            parent_id=int(branch_ids[termination_idx].item()),
        )

        branch_ids[path_vertices_idx] = branch_id
        branch_ids[idx_points] = branch_id

        branch_id += 1

    return branches
