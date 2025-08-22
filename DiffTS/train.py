import os
import subprocess
from collections.abc import MutableMapping
from os import environ, makedirs
from os.path import abspath, dirname, join

import click
import MinkowskiEngine as ME
import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import DiffTS.datasets.datasets as datasets
import DiffTS.models.models as models


def set_deterministic():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    seed_everything(42, workers=True)

def nested_update(d: MutableMapping, keys: list[str], value):
    """Recursively update nested dictionaries based on key list."""
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value

def parse_unknown_options(unknown_args):
    config_updates = {}
    i = 0
    while i < len(unknown_args):
        key = unknown_args[i]
        if not key.startswith('--'):
            raise click.ClickException(f"Invalid argument format: {key}")
        key = key[2:]  # remove leading --

        # Handle flags with values
        if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--'):
            value = unknown_args[i + 1]
            i += 2
        else:
            value = True  # flag with no value becomes boolean True
            i += 1

        # Type conversion
        if isinstance(value, str):
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass

        # Parse dotted keys into nested structure
        keys = key.split('.')
        nested_update(config_updates, keys, value)

    return config_updates

def merge_dicts(base: dict, overrides: dict):
    """Recursively merge overrides into base config."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_dicts(base[key], value)
        else:
            base[key] = value
            
def load_config(config_path, params):
    default_config_path = join(dirname(abspath(__file__)),'config/default_config.yaml')
    cfg = yaml.safe_load(open(default_config_path))
    exp_cfg = yaml.safe_load(open(config_path))
    merge_dicts(cfg, exp_cfg)
        
    cfg['git_commit_version'] = str(subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']).strip())

    updates = parse_unknown_options(list(params))
    print("param overrides", updates)
    merge_dicts(cfg, updates)
    if cfg['data']['debug_vis']:
        cfg['train']['num_workers'] = 0
    return cfg

@click.command(context_settings=dict(ignore_unknown_options=True))
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default='config.yaml')
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)
@click.option('--logdir', '-l', type=str, help='path to log directory', default="/logs")
@click.option('--test', '-t', is_flag=True, help='test mode')
@click.argument('params', nargs=-1, type=click.UNPROCESSED)
def main(config, weights, checkpoint, logdir, test, params):
    set_deterministic()
    print('Deterministic mode ON')
    if checkpoint == "None":
        checkpoint = None
    cfg = load_config(config, params)

    print("Starting experiment ", cfg['experiment']['id'])
    # overwrite the data path in case we have defined in the env variables
    if environ.get('TRAIN_DATABASE'):
        cfg['data']['data_dir'] = environ.get('TRAIN_DATABASE')

    #Load data and model
    if weights is None:
        model = models.DiffusionPoints(cfg)
    else:
        model = models.DiffusionPoints.load_from_checkpoint(weights, hparams=cfg)
        print("Used params: ", model.hparams)
        
    data = datasets.dataloaders[cfg['data']['dataloader']](cfg)
    #Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_saver = ModelCheckpoint(
                                 filename=cfg['experiment']['id']+'_{epoch:02d}',
                                 save_top_k=-1
                                 )

    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(logdir,'experiments/'+cfg['experiment']['id']),
                                             default_hp_metric=False)
    
    #Save git diff to keep track of all changes
    makedirs(tb_logger.log_dir, exist_ok=True)
    with open(f'{tb_logger.log_dir}/project.diff', 'w+') as diff_file:
        repo_diff = subprocess.run(['git', 'diff'], stdout=subprocess.PIPE)
        diff_file.write(repo_diff.stdout.decode('utf-8'))
    #Setup trainer
    if torch.cuda.device_count() > 1:
        cfg['train']['n_gpus'] = torch.cuda.device_count()
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        trainer = Trainer(gpus=cfg['train']['n_gpus'],
                          logger=tb_logger,
                          log_every_n_steps=100,
                          resume_from_checkpoint=checkpoint,
                          max_epochs= cfg['train']['max_epoch'],
                          callbacks=[lr_monitor, checkpoint_saver],
                          strategy="ddp",
                          num_sanity_val_steps=1 if cfg['data']['test_w_uncond'] else 0,
                          )
    else:
        trainer = Trainer(gpus=cfg['train']['n_gpus'],
                          logger=tb_logger,
                          log_every_n_steps=100,
                          resume_from_checkpoint=checkpoint,
                          max_epochs= cfg['train']['max_epoch'],
                          callbacks=[lr_monitor, checkpoint_saver],
                          num_sanity_val_steps=1 if cfg['data']['test_w_uncond'] else 0,
                        #   profiler="advanced"
                          )
    # Train!
    if test:
        print('TESTING MODE')
        trainer.test(model, data)
    else:
        print('TRAINING MODE')
        trainer.fit(model, data)

if __name__ == "__main__":
    main(auto_envvar_prefix='PLS')
