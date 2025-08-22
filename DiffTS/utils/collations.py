import numpy as np
import MinkowskiEngine as ME
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d

class SkeletonCollation:
    def __init__(self, mode='diffusion'):
        self.mode = mode
        return

    def __call__(self, data):
        # "transpose" the  batch(pt, ptn) to batch(pt), batch(ptn)
        batch = list(zip(*data))

        return {'pcd_nodes': batch[0],
                'full_pcd_nodes': batch[1],
            'node_parent_ids': batch[2],
            'pcd_conditioning_pts': batch[3],
            'scan_point_classes': batch[4],
            'scan_point_colors': batch[5],
            'filename': batch[6],
            'parent_ids': batch[7],
            'normalization_factors': batch[8],
        }
