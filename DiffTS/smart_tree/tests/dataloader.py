import time

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm



@hydra.main(
    version_base=None,
    config_path=".",
    config_name="test-dataloader",
)
def main(cfg: DictConfig):
    train_dataloader = instantiate(
        cfg.data_loader, dataset=instantiate(cfg.dataset, mode="train")
    )

    start_time = time.time()
    for data in tqdm(train_dataloader):
        pass

    print(time.time() - start_time)


if __name__ == "__main__":
    main()
