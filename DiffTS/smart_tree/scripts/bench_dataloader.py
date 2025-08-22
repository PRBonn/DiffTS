import logging
import time

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
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

    while True:
        start = time.time()

        batches = get_batch(train_loader, device="cpu")
        for sparse_input, targets, mask, filenames in tqdm(batches):
            pass

        print(time.time() - start)


if __name__ == "__main__":
    main()
