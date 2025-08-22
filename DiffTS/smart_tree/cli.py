import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig



@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="pipeline",
)
def main(cfg: DictConfig):
    pipeline = instantiate(cfg.pipeline)

    if "path" in dict(cfg):
        pipeline.process_cloud(Path(cfg.path))

    elif "directory" in dict(cfg):
        for p in os.listdir(cfg.directory):
            pipeline.process_cloud(Path(f"{cfg.directory}/{p}"))

    else:
        print("Please supply a path or directory to point clouds.")


if __name__ == "__main__":
    main()
