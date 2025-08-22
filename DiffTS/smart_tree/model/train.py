import contextlib
import logging
import math
import os
from pathlib import Path
from typing import List

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb
from smart_tree.data_types.cloud import Cloud
from smart_tree.o3d_abstractions.camera import Renderer

from .helper import get_batch, model_output_to_labelled_clds
from .tracker import Tracker


def train_epoch(
    data_loader,
    model,
    optimizer,
    loss_fn,
    fp16=False,
    scaler=None,
    device=torch.device("cuda"),
):
    tracker = Tracker()

    for sp_input, targets, mask, fn in tqdm(
        get_batch(data_loader, device, fp16),
        desc="Batch",
        leave=False,
        total=math.ceil(len(data_loader) / data_loader.batch_size),
    ):
        preds = model.forward(sp_input)

        loss = loss_fn(preds, targets, mask)

        if fp16:
            assert sum(loss.values()).dtype is torch.float32
            scaler.scale(sum(loss.values())).backward()
            scaler.step(optimizer)
            scaler.update()
            scale = scaler.get_scale()
        else:
            (sum(loss.values())).backward()
            optimizer.step()

        optimizer.zero_grad()
        tracker.update(loss)

    return tracker, scaler


@torch.no_grad()
def eval_epoch(
    data_loader,
    model,
    loss_fn,
    fp16=False,
    device=torch.device("cuda"),
):
    tracker = Tracker()
    model.eval()

    for sp_input, targets, mask, fn in tqdm(
        get_batch(data_loader, device, fp16),
        desc="Evaluating",
        leave=False,
        total=math.ceil(len(data_loader) / data_loader.batch_size),
    ):
        preds = model.forward(sp_input)
        loss = loss_fn(preds, targets, mask)
        tracker.update(loss)
    model.train()

    return tracker


@torch.no_grad()
def capture_epoch(
    data_loader,
    model,
    cmap,
    fp16=False,
    device=torch.device("cuda"),
):
    model.eval()
    captures = []

    for sp_input, targets, mask, fn in tqdm(
        get_batch(data_loader, device, fp16),
        desc="Capturing Outputs",
        leave=False,
        total=math.ceil(len(data_loader) / data_loader.batch_size),
    ):
        model_output = model.forward(sp_input)

        labelled_clouds = model_output_to_labelled_clds(
            sp_input,
            model_output,
            cmap,
            fn,
        )

    model.train()
    return labelled_clouds


@torch.no_grad()
def capture_clouds(
    data_loader,
    model,
    cmap,
    fp16=False,
    device=torch.device("cuda"),
) -> List[Cloud]:
    model.eval()
    clouds = []

    for sp_input, targets, mask, filenames in tqdm(
        get_batch(data_loader, device, fp16),
        desc="Capturing Outputs",
        leave=False,
        total=math.ceil(len(data_loader) / data_loader.batch_size),
    ):
        model_output = model.forward(sp_input)
        clouds.extend(
            model_output_to_labelled_clds(
                sp_input,
                model_output,
                cmap,
                filenames,
            )
        )

    model.train()
    return clouds


def capture_and_log(loader, model, epoch, wandb_run, cfg):
    clouds = capture_clouds(
        loader,
        model,
        cfg.cmap,
        fp16=cfg.fp16,
    )

    for cloud in tqdm(clouds, desc="Uploading Clouds", leave=False):
        seg_cloud = cloud.to_o3d_seg_cld(np.asarray(cfg.cmap))
        xyz_rgb = np.concatenate(
            (np.asarray(seg_cloud.points), np.asarray(seg_cloud.colors) * 255), -1
        )
        wandb_run.log(
            {f"{Path(cloud.filename).stem}": wandb.Object3D(xyz_rgb)},
            step=epoch,
        )


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="training.yaml",
)
def main(cfg: DictConfig):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    log = logging.getLogger(__name__)

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=[True | False]),
    )
    run_dir = HydraConfig.get().runtime.output_dir
    run_name = wandb.run.name
    log.info(f"Directory : {run_dir}")
    log.info(f"Machine: {os.uname()[1]}")

    renderer = Renderer(960, 540)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader = instantiate(cfg.train_data_loader)
    val_loader = instantiate(cfg.validation_data_loader)
    test_loader = instantiate(cfg.test_data_loader)

    log.info(f"Train Dataset Size: {len(train_loader.dataset)}")
    log.info(f"Validation Dataset Size: {len(val_loader.dataset)}")
    log.info(f"Test Dataset Size: {len(test_loader.dataset)}")

    # Model
    model = instantiate(cfg.model).to(device).train()
    torch.save(model, f"{run_dir}/{run_name}_model.pt")

    # Optimizer / Scheduler
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
    loss_fn = instantiate(cfg.loss_fn)

    # FP-16
    amp_ctx = torch.cuda.amp.autocast() if cfg.fp16 else contextlib.nullcontext()
    scaler = torch.cuda.amp.grad_scaler.GradScaler()

    epochs_no_improve = 0
    best_val_loss = torch.inf

    # Epochs
    for epoch in tqdm(range(0, cfg.num_epoch), leave=True, desc="Epoch"):
        with amp_ctx:
            training_tracker, scaler = train_epoch(
                train_loader,
                model,
                optimizer,
                loss_fn,
                scaler=scaler,
                fp16=cfg.fp16,
            )

            val_tracker = eval_epoch(
                val_loader,
                model,
                loss_fn,
                fp16=cfg.fp16,
            )

            test_tracker = eval_epoch(
                test_loader,
                model,
                loss_fn,
                fp16=cfg.fp16,
            )

            if (epoch + 1) % cfg.capture_output == 0:
                capture_and_log(test_loader, model, epoch, wandb.run, cfg)
                capture_and_log(val_loader, model, epoch, wandb.run, cfg)

        scheduler.step(val_tracker.total_loss) if cfg.lr_decay else None

        # Save Best Model
        if val_tracker.total_loss < best_val_loss:
            epochs_no_improve = 0
            best_val_loss = val_tracker.total_loss
            wandb.run.summary["Best Test Loss"] = best_val_loss
            torch.save(model.state_dict(), f"{run_dir}/{run_name}_model_weights.pt")
            log.info(f"Weights Saved at epoch: {epoch}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve == cfg.early_stop_epoch and cfg.early_stop:
            log.info("Training Ended (Evaluation Test Score Not Improving)")
            break

        # log onto wandb...
        training_tracker.log("Training", epoch)
        val_tracker.log("Validation", epoch)


if __name__ == "__main__":
    main()
