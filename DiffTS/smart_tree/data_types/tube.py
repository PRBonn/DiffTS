from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torchtyping import TensorType


@dataclass
class Tube:
    a: TensorType["N", 3]  # Start Point 3
    b: TensorType["N", 3]  # End Point 3
    r1: float  # Start Radius
    r2: float  # End Radius

    def to_torch(self, device=torch.device("cuda:0")):
        self.a = torch.from_numpy(self.a).float().to(device)
        self.b = torch.from_numpy(self.b).float().to(device)
        self.r1 = torch.from_numpy(self.r1).float().to(device)
        self.r2 = torch.from_numpy(self.r2).float().to(device)

    def to_numpy(self):
        self.a = self.a.cpu().detach().numpy()
        self.b = self.b.cpu().detach().numpy()
        self.r1 = self.r1.cpu().detach().numpy()
        self.r2 = self.r2.cpu().detach().numpy()


@dataclass
class CollatedTube:
    a: TensorType["N", 3]  # Nx3
    b: TensorType["N", 3]  # Nx3
    r1: TensorType["N", 1]  # N
    r2: TensorType["N", 1]  # N

    def to_gpu(self, device=torch.device("cuda")):
        self.a = self.a.to(device)
        self.b = self.b.to(device)
        self.r1 = self.r1.to(device)
        self.r2 = self.r2.to(device)


def collate_tubes(tubes: List[Tube]) -> CollatedTube:
    a = torch.cat([tube.a for tube in tubes]).reshape(-1, 3)
    b = torch.cat([tube.b for tube in tubes]).reshape(-1, 3)

    r1 = torch.cat([tube.r1 for tube in tubes]).reshape(1, -1)
    r2 = torch.cat([tube.r2 for tube in tubes]).reshape(1, -1)

    return CollatedTube(a, b, r1, r2)


def sample_tubes(tubes: List[Tube], spacing):
    pts, radius = [], []

    for i, tube in enumerate(tubes):
        v = tube.b - tube.a
        length = np.linalg.norm(v)

        direction = v / length
        num_points = np.ceil(length / spacing)

        if int(num_points) > 0.0:
            spaced_points = np.arange(
                0, float(length), step=float(length / num_points)
            ).reshape(-1, 1)
            lin_radius = np.linspace(
                tube.r1, tube.r2, spaced_points.shape[0], dtype=float
            )

            pts.append(tube.a + direction * spaced_points)
            radius.append(lin_radius)

    return np.concatenate(pts, axis=0), np.concatenate(radius, axis=0)
