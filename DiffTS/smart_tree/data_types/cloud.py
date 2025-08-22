from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchtyping import TensorType
from typeguard import typechecked

from ..o3d_abstractions.geometries import o3d_cloud, o3d_lines_between_clouds
from ..o3d_abstractions.visualizer import o3d_viewer
from ..util.misc import voxel_downsample
from ..util.queries import skeleton_to_points


@typechecked
@dataclass
class Cloud:
    xyz: TensorType["N", 3]
    rgb: Optional[TensorType["N", 3]] = None
    medial_vector: Optional[TensorType["N", 3]] = None
    branch_direction: Optional[TensorType["N", 3]] = None
    branch_ids: Optional[TensorType["N", 1]] = None
    class_l: Optional[TensorType["N", 1]] = None
    filename: Optional[Path] = None

    def __len__(self):
        return self.xyz.shape[0]

    def __str__(self):
        return f"{'*' * 80}\nCloud with {self.xyz.shape[0]} Points.\nMin: {torch.min(self.xyz, 0)[0]}\nMax: {torch.max(self.xyz, 0)[0]}\nDevice:{self.xyz.device}\n{'*' * 80}\n"

    def paint(self, colour=[1, 0, 0]):
        self.rgb = torch.tensor([colour]).expand(self.__len__, -1)

    def to_o3d_cld(self):
        cpu_cld = self.cpu()
        if not hasattr(cpu_cld, "rgb"):
            cpu_cld.paint()
        return o3d_cloud(cpu_cld.xyz, colours=cpu_cld.rgb)

    def to_o3d_seg_cld(self, cmap: np.ndarray = np.array([[1, 0, 0], [0, 1, 0]])):
        cpu_cld = self.cpu()
        colours = cmap[cpu_cld.class_l.view(-1).int()]
        return o3d_cloud(cpu_cld.xyz, colours=colours)

    def to_o3d_trunk_cld(self):
        cpu_cld = self.cpu()
        min_branch_id = cpu_cld.branch_ids[0]
        return cpu_cld.filter(cpu_cld.branch_ids == min_branch_id).to_o3d_cld()

    def to_o3d_branch_cld(self):
        cpu_cld = self.cpu()
        min_branch_id = cpu_cld.branch_ids[0]
        return cpu_cld.filter(cpu_cld.branch_ids != min_branch_id).to_o3d_cld()

    def to_o3d_medial_vectors(self, cmap=None):
        cpu_cld = self.cpu()
        medial_cloud = o3d_cloud(cpu_cld.xyz + cpu_cld.medial_vector)
        return o3d_lines_between_clouds(cpu_cld.to_o3d_cld(), medial_cloud)

    def to_o3d_branch_directions(self, scale=0.1, cmap=None):
        cpu_cld = self.cpu()
        branch_dir_cloud = o3d_cloud(cpu_cld.xyz + (cpu_cld.branch_direction * scale))
        return o3d_lines_between_clouds(cpu_cld.to_o3d_cld(), branch_dir_cloud)

    # def to_wandb_seg_cld(self, cmap: np.ndarray = np.array([[1, 0, 0], [0, 1, 0]])):

    def filter(cloud: Cloud, mask) -> Cloud:
        mask = mask.to(cloud.xyz.device)
        xyz = cloud.xyz[mask]
        rgb = cloud.rgb[mask] if cloud.rgb is not None else None

        medial_vector = (
            cloud.medial_vector[mask] if cloud.medial_vector is not None else None
        )
        branch_direction = (
            cloud.branch_direction[mask] if cloud.branch_direction is not None else None
        )
        class_l = cloud.class_l[mask] if cloud.class_l is not None else None
        branch_ids = cloud.branch_ids[mask] if cloud.branch_ids is not None else None
        filename = cloud.filename if cloud.filename is not None else None

        return Cloud(
            xyz=xyz,
            rgb=rgb,
            medial_vector=medial_vector,
            branch_direction=branch_direction,
            class_l=class_l,
            branch_ids=branch_ids,
            filename=filename,
        )

    def filter_by_class(self, classes):
        classes = torch.tensor(classes, device=self.class_l.device)
        mask = torch.isin(
            self.class_l,
            classes,
        )
        return self.filter(mask.view(-1))

    def filter_by_skeleton(skeleton: TreeSkeleton, cloud: Cloud, threshold=1.1):
        distances, radii, vectors_ = skeleton_to_points(cloud, skeleton, chunk_size=512)
        mask = distances < radii * threshold
        return filter(cloud, mask)

    def pin_memory(self):
        xyz = self.xyz.pin_memory()
        rgb = self.rgb.pin_memory() if self.rgb is not None else None
        medial_vector = (
            self.medial_vector.pin_memory() if self.medial_vector is not None else None
        )
        branch_direction = (
            self.branch_direction.pin_memory()
            if self.branch_direction is not None
            else None
        )
        class_l = self.class_l.pin_memory() if self.class_l is not None else None
        branch_ids = (
            self.branch_ids.pin_memory() if self.branch_ids is not None else None
        )

        return Cloud(
            xyz=xyz,
            rgb=rgb,
            medial_vector=medial_vector,
            branch_direction=branch_direction,
            class_l=class_l,
            branch_ids=branch_ids,
        )

    def cpu(self):
        return self.to_device(torch.device("cpu"))

    def to_device(self, device):
        xyz = self.xyz.to(device)
        rgb = self.rgb.to(device) if self.rgb is not None else None
        medial_vector = (
            self.medial_vector.to(device) if self.medial_vector is not None else None
        )
        branch_direction = (
            self.branch_direction.to(device)
            if self.branch_direction is not None
            else None
        )
        class_l = self.class_l.to(device) if self.class_l is not None else None
        branch_ids = self.branch_ids.to(device) if self.branch_ids is not None else None
        filename = self.filename if self.filename is not None else None

        return Cloud(
            xyz=xyz,
            rgb=rgb,
            medial_vector=medial_vector,
            branch_direction=branch_direction,
            class_l=class_l,
            branch_ids=branch_ids,
            filename=filename,
        )

    def cat(self):
        return torch.cat(
            (
                self.xyz,
                self.rgb,
            ),
            1,
        )

    def view(self, cmap=[]):
        if cmap == []:
            cmap = np.random.rand(self.number_classes, 3)

        cpu_cld = self.cpu()
        geoms = []

        geoms.append(cpu_cld.to_o3d_cld())
        if cpu_cld.class_l != None:
            geoms.append(cpu_cld.to_o3d_seg_cld(cmap))

        if cpu_cld.medial_vector != None:
            projected = o3d_cloud(cpu_cld.xyz + cpu_cld.medial_vector, colour=(1, 0, 0))
            geoms.append(projected)
            geoms.append(o3d_lines_between_clouds(cpu_cld.to_o3d_cld(), projected))

        o3d_viewer(geoms)

    def voxel_down_sample(self, voxel_size):
        idx = voxel_downsample(self.xyz, voxel_size)
        return self.filter(idx)

    def scale(self, factor):
        return Cloud(self.xyz * factor, self.rgb)

    def translate(self, xyz):
        return Cloud(self.xyz + xyz.to(self.xyz.device), self.rgb)

    def rotate(self, rot_mat):
        rot_mat = rot_mat.to(self.xyz.dtype)
        return Cloud(torch.matmul(self.xyz, rot_mat.to(self.xyz.device)), self.rgb)

    @property
    def root_idx(self) -> int:
        return torch.argmin(self.xyz[:, 1]).item()

    @property
    def number_classes(self) -> int:
        if not hasattr(self, "class_l"):
            return 1
        return torch.max(self.class_l).item() + 1

    @property
    def max_xyz(self) -> torch.Tensor:
        return torch.max(self.xyz, 0)[0]

    @property
    def min_xyz(self) -> torch.Tensor:
        return torch.min(self.xyz, 0)[0]

    @property
    def bbox(self) -> tuple[torch.Tensor, torch.Tensor]:
        # defined by centre coordinate, x/2, y/2, z/2
        dimensions = (self.max_xyz - self.min_xyz) / 2
        centre = self.min_xyz + dimensions
        return centre, dimensions

    @property
    def medial_pts(self) -> torch.Tensor:
        return self.xyz + self.medial_vector

    @staticmethod
    def from_numpy(**kwargs) -> "Cloud":
        torch_kwargs = {}

        for key, value in kwargs.items():
            if key in [
                "xyz",
                "rgb",
                "medial_vector",
                "branch_direction",
                "branch_ids",
                "class_l",
            ]:
                torch_kwargs[key] = torch.tensor(value).float()

            """ SUPPORT LEGACY NPZ -> Remove in Future..."""
            if key in ["vector"]:
                torch_kwargs["medial_vector"] = torch.tensor(value)

        return Cloud(**torch_kwargs)

    @property
    def radius(self) -> torch.Tensor:
        return self.medial_vector.pow(2).sum(1).sqrt()

    @property
    def direction(self) -> torch.Tensor:
        return F.normalize(self.medial_vector)

    @staticmethod
    def from_o3d_cld(cld) -> Cloud:
        return Cloud.from_numpy(xyz=np.asarray(cld.points), rgb=np.asarray(cld.colors))
