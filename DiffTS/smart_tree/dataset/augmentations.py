from abc import ABC, abstractmethod
from typing import Sequence

import torch
from beartype import beartype

from smart_tree.data_types.cloud import Cloud
from smart_tree.util.maths import euler_angles_to_rotation


class Augmentation(ABC):
    @abstractmethod
    def __call__(self, cloud: Cloud) -> Cloud:
        pass


class Scale(Augmentation):
    def __init__(self, min_scale=0.9, max_scale=1.1):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, cloud: Cloud) -> Cloud:
        t = torch.rand(1, device=cloud.xyz.device) * (self.max_scale - self.min_scale)
        return cloud.scale(t + self.min_scale)


class FixedRotate(Augmentation):
    def __init__(self, xyz):
        self.xyz = xyz

    def __call__(self, cloud: Cloud) -> Cloud:
        self.rot_mat = euler_angles_to_rotation(
            torch.tensor(self.xyz, device=cloud.xyz.device)
        ).float()
        return cloud.rotate(self.rot_mat)


class CentreCloud(Augmentation):
    def __call__(self, cloud: Cloud) -> Cloud:
        centre, (x, y, z) = cloud.bbox
        return cloud.translate(-centre + torch.tensor([0, y, 0], device=centre.device))


class VoxelDownsample(Augmentation):
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def __call__(self, cloud: Cloud) -> Cloud:
        return cloud.voxel_down_sample(self.voxel_size)


class FixedTranslate(Augmentation):
    def __init__(self, xyz):
        self.xyz = torch.tensor(xyz)

    def __call__(self, cloud: Cloud) -> Cloud:
        return cloud.translate(self.xyz)


class RandomCrop(Augmentation):
    def __init__(self, max_x, max_y, max_z):
        self.max_translation = torch.tensor([max_x, max_y, max_z])

    def __call__(self, cloud):
        offset = (
            torch.rand(3, device=cloud.xyz.device) - 0.5
        ) * self.max_translation.to(device=cloud.xyz.device)

        p = cloud.xyz + offset
        mask = torch.logical_and(p >= cloud.min_xyz, p <= cloud.max_xyz).all(dim=1)

        return cloud.filter(mask)


class RandomCubicCrop(Augmentation):
    def __init__(self, size):
        self.size = size

    def __call__(self, cloud):
        random_pt = cloud.xyz[torch.randint(0, cloud.xyz.shape[0], (1,))]
        min_corner = random_pt - self.size / 2
        max_corner = random_pt + self.size / 2

        mask = torch.logical_and(
            cloud.xyz >= min_corner,
            cloud.xyz <= max_corner,
        ).all(dim=1)

        return cloud.filter(mask)


class RandomDropout(Augmentation):
    def __init__(self, max_drop_out):
        self.max_drop_out = max_drop_out

    def __call__(self, cloud: Cloud) -> Cloud:
        num_indices = int(
            (1.0 - (self.max_drop_out * torch.rand(1, device=cloud.xyz.device)))
            * cloud.xyz.shape[0]
        )

        indices = torch.randint(
            high=cloud.xyz.shape[0], size=(num_indices, 1), device=cloud.xyz.device
        ).squeeze(1)
        return cloud.filter(indices)


class AugmentationPipeline(Augmentation):
    @beartype
    def __init__(self, augmentations: Sequence[Augmentation]):
        self.augmentations = augmentations

    def __call__(self, cloud):
        for augmentation in self.augmentations:
            cloud = augmentation(cloud)
        return cloud
