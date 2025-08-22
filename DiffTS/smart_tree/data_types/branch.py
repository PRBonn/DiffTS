from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import open3d as o3d
import torch
from torchtyping import TensorType
from typeguard import typechecked

from smart_tree.o3d_abstractions.geometries import o3d_path, o3d_tube_mesh

from .tube import Tube


@typechecked
@dataclass
class BranchSkeleton:
    _id: int
    parent_id: int
    xyz: TensorType["N", 3]
    radii: TensorType["N", 1]
    child_id: Optional[int] = None

    def __post_init__(self) -> None:
        self.colour = np.random.rand(3)

    def __len__(self) -> np.array:
        return self.xyz.shape[0]

    def __str__(self):
        return f" ID: {self._id} \
                  Points: {self.xyz} \
                  Radii {self.radii}"

    def to_o3d_lineset(self, colour=(0, 0, 0)) -> o3d.geometry.LineSet:
        return o3d_path(self.xyz, colour)

    def to_o3d_tube(self) -> o3d.geometry.TriangleMesh:
        return o3d_tube_mesh(self.xyz.numpy(), self.radii.numpy(), self.colour)

    def to_tubes(self, colour=(1, 0, 0)) -> List[Tube]:
        a_, b_, r1_, r2_ = (
            self.xyz[:-1],
            self.xyz[1:],
            self.radii[:-1],
            self.radii[1:],
        )
        return [Tube(a, b, r1, r2) for a, b, r1, r2 in zip(a_, b_, r1_, r2_)]

    def filter(self, mask) -> BranchSkeleton:
        return BranchSkeleton(
            self._id,
            self.parent_id,
            self.xyz[mask],
            self.radii[mask],
            self.child_id,
        )

    @property
    def length(self) -> TensorType[1]:
        return (self.xyz[1:] - self.xyz[:-1]).norm(dim=1).sum()

    @property
    def initial_radius(self) -> TensorType[1]:
        return torch.max(self.radii[0], self.radii[-1])

    @property
    def biggest_radius_idx(self) -> TensorType[1]:
        return torch.argmax(self.radii)

    @property
    def biggest_radius(self) -> TensorType[1]:
        return torch.max(self.radii)
