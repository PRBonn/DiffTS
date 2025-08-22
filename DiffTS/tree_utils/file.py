import numpy as np
import open3d as o3d

from pathlib import Path
from typing import Tuple


from .cloud import Cloud
from .tree import TreeSkeleton
from .branch import BranchSkeleton


def unpackage_data(data: dict) -> Tuple[Cloud, TreeSkeleton]:
    tree_id = data["tree_id"]
    branch_id = data["branch_id"]
    branch_parent_id = data["branch_parent_id"]
    skeleton_xyz = data["skeleton_xyz"].astype(np.float32)
    skeleton_radii = data["skeleton_radii"].astype(np.float32)
    sizes = data["branch_num_elements"]
    medial_vector = data.get("medial_vector", data.get("vector", None))

    cld = Cloud(
        xyz=data["xyz"],
        rgb=data["rgb"],
        class_l=data["class_l"],
        medial_vector=medial_vector,
    )

    offsets = np.cumsum(np.append([0], sizes))

    branch_idx = [np.arange(size) + offset for size, offset in zip(sizes, offsets)]
    branches = {}

    for idx, _id, parent_id in zip(branch_idx, branch_id, branch_parent_id):
        branches[_id] = BranchSkeleton(
            _id, parent_id, skeleton_xyz[idx], skeleton_radii[idx]
        )

    return cld, TreeSkeleton(tree_id, branches)


def load_data_npz(path: Path) -> Tuple[Cloud, TreeSkeleton]:
    return unpackage_data(np.load(str(path)))


def load_o3d_cloud(path: Path):
    return o3d.io.read_point_cloud(str(path))
