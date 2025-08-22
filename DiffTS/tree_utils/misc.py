import numpy as np
import torch

from typing import List

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def to_torch(numpy_arrays: List[np.array], device=torch.device("cpu")):
  return [torch.from_numpy(np_arr).float().to(device) for np_arr in numpy_arrays]