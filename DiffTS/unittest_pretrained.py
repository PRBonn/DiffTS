import click
from os.path import join, dirname, abspath
from os import environ, makedirs
import subprocess

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
# from lightning.pytorch.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import numpy as np
import torch
import yaml
import os

import MinkowskiEngine as ME

import DiffTS.datasets.datasets as datasets
import DiffTS.models.models as models
from collections.abc import MutableMapping

from DiffTS.train import merge_dicts, set_deterministic, load_config

def run_pipeline(cfg_path: str, weights_path: str, param_overrides):
    set_deterministic()
    cfg = load_config(cfg_path, param_overrides)
    
    cfg["data"]["save_pcds"] = False

    model = models.DiffusionPoints.load_from_checkpoint(weights_path, hparams=cfg)

    data = datasets.dataloaders[cfg['data']['dataloader']](cfg)

    trainer = Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        gpus=0 if not torch.cuda.is_available() else 1,
        num_sanity_val_steps=0,
        limit_test_batches=1,
    )

    result = trainer.test(model, data)
    return result

def print_red(text):
    print(f"\033[91m{text}\033[0m")

if __name__ == "__main__":
    test_models = [
        ("pretrained_models/orchard_model.ckpt", "pretrained_models/orchard_model.yaml"),
        ("pretrained_models/syntheticTrees_model.ckpt", "pretrained_models/syntheticTrees_model.yaml"),
        ("pretrained_models/treenet3d_model.ckpt", "pretrained_models/treenet3d_model.yaml"),
    ]
    expected_results = [
        {'test/node_chamfer_dist_epoch': 0.07484608216032185, 'test/node_chamfer_dist_p2g_epoch': 0.06457810739269432, 'test/node_chamfer_dist_g2p_epoch': 0.08511405692794938, 
         'test/nn_skel_chamfer_dist_epoch': 0.0743388254043295, 'test/nn_skel_chamfer_dist_p2g_epoch': 0.06534635249924407, 'test/nn_skel_chamfer_dist_g2p_epoch': 0.08333129830941494, 
         'test/pp_skel_chamfer_dist_epoch': 0.07454763629620292, 'test/pp_skel_chamfer_dist_p2g_epoch': 0.0656626991318893, 'test/pp_skel_chamfer_dist_g2p_epoch': 0.08343257346051657, 
         },
        {'test/node_chamfer_dist_epoch': 0.043739641861931855, 'test/node_chamfer_dist_p2g_epoch': 0.04316573931668935, 'test/node_chamfer_dist_g2p_epoch': 0.04431354440717437, 
         'test/nn_skel_chamfer_dist_epoch': 0.040640544902262654, 'test/nn_skel_chamfer_dist_p2g_epoch': 0.04091229175563769, 'test/nn_skel_chamfer_dist_g2p_epoch': 0.04036879804888763, 
         'test/pp_skel_chamfer_dist_epoch': 0.041918621055770815, 'test/pp_skel_chamfer_dist_p2g_epoch': 0.043206291040137255, 'test/pp_skel_chamfer_dist_g2p_epoch': 0.040630951071404374, 
         },
        {'test/node_chamfer_dist_epoch': 0.11258536680518247, 'test/node_chamfer_dist_p2g_epoch': 0.11043244130918713, 'test/node_chamfer_dist_g2p_epoch': 0.11473829230117778, 
         'test/nn_skel_chamfer_dist_epoch': 0.09881631326239401, 'test/nn_skel_chamfer_dist_p2g_epoch': 0.10348177580236381, 'test/nn_skel_chamfer_dist_g2p_epoch': 0.09415085072242423, 
         'test/pp_skel_chamfer_dist_epoch': 0.10916930715507077, 'test/pp_skel_chamfer_dist_p2g_epoch': 0.09371843592910936, 'test/pp_skel_chamfer_dist_g2p_epoch': 0.12462017838103218, 
         },
    ]
    for (weights_path, cfg_path), expected_result in zip(test_models, expected_results):
        print_red(f"Testing model: {weights_path}")
        result = run_pipeline(cfg_path, weights_path, {})
        for key, value in expected_result.items():
            assert key in result[0], f"Key {key} not found in result"
            assert np.abs(result[0][key] - value) < 2e-3, f"Expected {value} for {key}, got {result[0][key]}, which is off by {np.abs(result[0][key] - value)}"
        print_red(f"Model {weights_path} passed all tests.")
    print_red("###################  All models passed the tests successfully. ###################")
