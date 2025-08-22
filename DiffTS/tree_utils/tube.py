import numpy as np

from typing import List
from dataclasses import dataclass


@dataclass
class Tube:
    a: np.array  # Start Point 3
    b: np.array  # End Point 3
    r1: float  # Start Radius
    r2: float  # End Radius


@dataclass
class CollatedTube:
    a: np.array  # Nx3
    b: np.array  # Nx3
    r1: np.array  # N
    r2: np.array  # N


def collate_tubes(tubes: List[Tube]) -> CollatedTube:

    a = np.concatenate([tube.a for tube in tubes]).reshape(-1, 3)
    b = np.concatenate([tube.b for tube in tubes]).reshape(-1, 3)

    r1 = np.asarray([tube.r1 for tube in tubes]).reshape(1, -1)
    r2 = np.asarray([tube.r2 for tube in tubes]).reshape(1, -1)

    return CollatedTube(a, b, r1, r2)
