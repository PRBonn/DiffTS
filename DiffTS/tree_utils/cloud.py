import numpy as np

import torch

from dataclasses import dataclass
from .o3d_abstractions import o3d_cloud, o3d_lines_between_clouds


@dataclass
class Cloud:
    xyz: np.array
    rgb: np.array
    class_l: np.array = None
    medial_vector: np.array = None

    def __len__(self):
        return self.xyz.shape[0]

    def __str__(self):
        return f"Cloud with {self.xyz.shape[0]} points "

    def to_o3d_cloud(self):
        return o3d_cloud(self.xyz, colours=self.rgb)

    def to_o3d_cloud_labelled(self, cmap=None):
        if cmap is None:
            cmap = np.random.rand(self.number_classes, 3)

        return o3d_cloud(self.xyz, colours=cmap[self.class_l])

    def to_o3d_medial_vectors(self, cmap=None):
        medial_cloud = o3d_cloud(self.xyz + self.medial_vector)
        return o3d_lines_between_clouds(self.to_o3d_cloud(), medial_cloud)

    def to_device(self, device):
        if self.xyz is not None:
            self.xyz = (
                torch.from_numpy(self.xyz).to(device)
                if isinstance(self.xyz, np.ndarray)
                else self.xyz.to(device)
            )

        if self.rgb is not None:
            self.rgb = (
                torch.from_numpy(self.rgb).to(device)
                if isinstance(self.rgb, np.ndarray)
                else self.rgb.to(device)
            )

        if self.class_l is not None:
            self.class_l = (
                torch.from_numpy(self.class_l).to(device)
                if isinstance(self.class_l, np.ndarray)
                else self.class_l.to(device)
            )

        if self.medial_vector is not None:
            self.medial_vector = (
                torch.from_numpy(self.medial_vector).to(device)
                if isinstance(self.medial_vector, np.ndarray)
                else self.medial_vector.to(device)
            )

    @property
    def number_classes(self):
        return torch.max(torch.tensor(self.class_l)) + 1
