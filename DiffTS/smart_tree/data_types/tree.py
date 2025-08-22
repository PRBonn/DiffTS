import pickle
from dataclasses import dataclass
from typing import Dict, List

import open3d as o3d
import torch
import torch.nn.functional as F
from torchtyping import TensorType
from tqdm import tqdm

from ..o3d_abstractions.geometries import (o3d_merge_linesets,
                                           o3d_merge_meshes)
from ..o3d_abstractions.visualizer import o3d_viewer
from ..util.misc import flatten_list, merge_dictionaries
from ..util.queries import pts_to_nearest_tube_gpu
from .branch import BranchSkeleton
from .tube import Tube


@dataclass
class TreeSkeleton:
    _id: int
    branches: Dict[int, BranchSkeleton]

    def __post_init__(self):
        self.colour = torch.rand(3)

    def __len__(self) -> int:
        return len(self.branches)

    def __str__(self) -> str:
        return f"Tree Skeleton ({self._id}) has {len(self)} branches..."

    def to_o3d_linesets(self) -> List[o3d.geometry.LineSet]:
        return [b.to_o3d_lineset() for b in self.branches.values()]

    def to_o3d_lineset(self, colour=(0, 0, 0)) -> o3d.geometry.LineSet:
        return o3d_merge_linesets(self.to_o3d_linesets(), colour=colour)
    
    def to_o3d_lineset_with_branch_ids(self) -> o3d.cuda.pybind.geometry.LineSet:
        return [[branch.to_o3d_lineset(), branch._id] for branch in self.branches.values()]

    def to_o3d_tubes(self) -> List[o3d.geometry.TriangleMesh]:
        return [b.to_o3d_tube() for b in self.branches.values()]

    def to_o3d_tube(self) -> o3d.geometry.TriangleMesh:
        return o3d_merge_meshes(self.to_o3d_tubes())

    def view(self):
        o3d_viewer([self.to_o3d_lineset(), self.to_o3d_tube()])

    def to_tubes(self) -> List[Tube]:
        return flatten_list([b.to_tubes() for b in self.branches.values()])

    def sample_skeleton(self, spacing):
        return sample_tubes(self.to_tubes(), spacing)

    def to_device(self, device):
        for branch_id, branch in self.branches():
            return BranchSkeleton(
                _id,
                parent_id,
                self.xyz.to(device),
                self.radii.to(device),
                child_id,
            )

    def point_to_skeleton(self):
        tubes = self.to_tubes()

        def point_to_tube(pt):
            return pts_to_nearest_tube_gpu(pt.reshape(-1, 3), tubes)

        return point_to_tube  # v, idx, _

    def repair(self):
        """skeletons are not connected between branches.
        this function connects the branches to their parent branches by finding
        the nearest point on the parent branch."""

        branch_ids = [branch._id for branch in self.branches.values()]

        for branch in self.branches.values():
            if branch.parent_id not in branch_ids:
                continue

            parent_branch = self.branches[branch.parent_id]
            tubes = parent_branch.to_tubes()

            v, idx, _ = pts_to_nearest_tube_gpu(branch.xyz[0].reshape(-1, 3), tubes)

            connection_pt = branch.xyz[0].reshape(-1, 3).cpu() + v[0].cpu()

            branch.xyz = torch.cat((connection_pt, branch.xyz))
            branch.radii = torch.cat((branch.radii[[0]], branch.radii))

    def prune(self, min_radius: float, min_length: float, root_id=None):
        """
        If a branch doesn't meet the initial radius threshold or length threshold we want to remove it and all
        it's predecessors...
        minimum_radius: some point of the branch must be above this to not remove the branch
        length: the total lenght of the branch must be greater than this point
        """
        root_id = min(self.branches.keys()) if root_id == None else root_id

        keep = {root_id: self.branches[root_id]}
        remove = {}

        for branch_id, branch in tqdm(
            self.branches.items(),
            leave=False,
            desc="Pruning Branches",
        ):
            if branch.parent_id not in keep and branch._id != root_id:
                remove[branch_id] = branch
            elif branch.length < min_length:
                remove[branch_id] = branch
            elif branch.initial_radius < min_radius:
                remove[branch_id] = branch
            else:
                keep[branch_id] = branch

        self.branches = keep
        return TreeSkeleton(0, remove)

    def smooth(self, kernel_size=5):
        """
        Smooths the skeleton radius.
        """
        kernel = torch.ones(1, 1, kernel_size) / kernel_size
        for branch in self.branches.values():
            if branch.radii.shape[0] > kernel_size:
                branch.radii = F.conv1d(
                    branch.radii.reshape(1, 1, -1),
                    kernel,
                    padding="same",
                ).reshape(-1)

    @property
    def length(self) -> TensorType[1]:
        return torch.sum(torch.tensor([b.length for b in self.branches.values()]))

    @property
    def biggest_radius_idx(self) -> TensorType[1]:
        return torch.argmax(self.radii)

    @property
    def key_branch_with_biggest_radius(self) -> TensorType[1]:
        """Returns the key of the branch with the biggest radius"""
        biggest_branch_radius = 0
        for key, branch in self.branches.items():
            if branch.biggest_radius > biggest_branch_radius:
                biggest_branch_radius = branch.biggest_radius
                biggest_branch_radius_key = key

        return biggest_branch_radius_key

    @property
    def max_branch_id(self):
        return max(self.branches.keys())


@dataclass
class DisjointTreeSkeleton:
    skeletons: List[TreeSkeleton]

    def prune(self, min_radius, min_length):
        self.skeletons[0].prune(
            min_radius=min_radius,
            min_length=min_length,
        )  # Can only prune the first skeleton as we don't know the root points for all the other skeletons...

    def repair(self):
        for skeleton in self.skeletons:
            skeleton.repair()

    def smooth(self, kernel_size=7):
        for skeleton in self.skeletons:
            skeleton.smooth(kernel_size=kernel_size)

    def to_o3d_lineset(self):
        return o3d_merge_linesets(
            [s.to_o3d_lineset().paint_uniform_color(s.colour) for s in self.skeletons]
        )

    def to_o3d_tube(self, colour=True):
        if colour:
            skeleton_tubes = [
                skel.to_o3d_tube().paint_uniform_color(skel.colour)
                for skel in self.skeletons
            ]
        else:
            skeleton_tubes = [s.to_o3d_tube() for s in self.skeletons]

        return o3d_merge_meshes(skeleton_tubes)

    def view(self):
        o3d_viewer([self.to_o3d_lineset(), self.to_o3d_tube()])

    def to_pickle(self, path):
        with open("disjoint_skeleton.pkl", "wb") as pickle_file:
            pickle.dump(self, pickle_file)

    @staticmethod
    def from_pickle(path):
        with open(f"{path}", "rb") as pickle_file:
            return pickle.load(pickle_file)


def connect(
    skeleton_1: TreeSkeleton,
    skeleton_1_child_branch_key: int,
    skeleton_1_child_vert_idx: int,
    skeleton_2: TreeSkeleton,
    skeleton_2_child_branch_key: int,
    skeleton_2_child_vert_idx: int,
) -> TreeSkeleton:
    # This is bundy, only visually gives appearance of connection...
    # Need to do some more processing to actually connect the skeletons...

    parent_branch = skeleton_1.branches[skeleton_1_parent_branch_key]
    child_branch = skeleton_2.branches[skeleton_2_child_branch_key]

    child_branch.parent_id = skeleton_1_parent_branch_key
    connection_pt = parent_branch.xyz[skeleton_1_parent_vert_idx]

    child_branch.xyz = torch.cat((connection_pt.unsqueeze(0), child_branch.xyz))
    child_branch.radii = torch.cat((child_branch.radii[[0]], child_branch.radii))

    for key, branch in skeleton_2.branches.items():
        branch._id += skeleton_1.max_branch_id

        if branch.parent_id != -1:
            branch.parent_id += skeleton_1.max_branch_id

    return TreeSkeleton(0, merge_dictionaries(skeleton_1.branches, skeleton_2.branches))
