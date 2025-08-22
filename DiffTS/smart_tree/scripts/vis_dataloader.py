import logging

import hydra
import numpy as np
import open3d as o3d
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from open3d_vis import render
from tqdm import tqdm

from smart_tree.model.helper import get_batch


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="vine-dataset",
)
def main(cfg: DictConfig):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    log = logging.getLogger(__name__)

    cfg = cfg.training
    torch.multiprocessing.set_start_method("spawn")

    train_loader = instantiate(cfg.train_data_loader)
    log.info(f"Train Dataset Size: {len(train_loader.dataset)}")

    batches = get_batch(train_loader, device="cpu")
    cmap = torch.from_numpy(np.array(cfg.cmap))

    for sparse_input, targets, mask, filenames in tqdm(batches):
        cloud_ids = sparse_input.indices[:, 0]
        coords = sparse_input.indices[:, 1:4]

        class_l = targets[:, -1].to(dtype=torch.long)

        for i, filename in enumerate(filenames):
            print("Filename:", filename)
            mask = cloud_ids == i
            labels = class_l[mask]

            xyz = coords[mask]
            class_colors = cmap[labels]

            boxes = render.boxes(xyz, xyz + 1, class_colors)
            o3d.visualization.draw([boxes])


if __name__ == "__main__":
    main()
