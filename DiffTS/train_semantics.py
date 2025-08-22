import os
import subprocess
from os import environ, makedirs
from os.path import abspath, dirname, join
from urllib.request import urlretrieve

import click
import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn as nn
import yaml
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
# from lightning.pytorch.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torchmetrics import JaccardIndex

import DiffTS.datasets.datasets as datasets
from DiffTS.models import minkunet_blocks


def set_deterministic():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    seed_everything(42, workers=True)

@click.command()
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'config/config_sem.yaml'))
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
def main(config, weights, checkpoint, logdir, test):
    set_deterministic()
    config = config.replace('config.yaml', 'config_sem.yaml')
    print('Deterministic mode ON')
    if checkpoint == "None":
        checkpoint = None
    default_config_path = join(dirname(abspath(__file__)),'config/default_config.yaml')
    cfg = yaml.safe_load(open(default_config_path))
    exp_cfg = yaml.safe_load(open(config))
    cfg.update(exp_cfg)
    cfg['git_commit_version'] = str(subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']).strip())

    print("Starting experiment ", cfg['experiment']['id'])
    # overwrite the data path in case we have defined in the env variables
    if environ.get('TRAIN_DATABASE'):
        cfg['data']['data_dir'] = environ.get('TRAIN_DATABASE')
        
    if not os.path.isfile('weights.pth'):
        print('Downloading weights...')
        urlretrieve("https://bit.ly/2O4dZrz", "weights.pth")

    
    #Load data and model
    if weights is None:
        model = SemanticSegmenter(cfg)
    else:
        model = SemanticSegmenter.load_from_checkpoint(weights, hparams=cfg)
        print(model.hparams)
    data = datasets.dataloaders[cfg['data']['dataloader']](cfg)
    #Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_saver = ModelCheckpoint(
            monitor="val/mean_iou",
            save_top_k=3,
            filename=cfg['experiment']['id']+'_{epoch:02d}',
            mode="max",
            save_last=True,
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
                          num_sanity_val_steps=0,
                          )
    else:
        trainer = Trainer(gpus=cfg['train']['n_gpus'],
                          logger=tb_logger,
                          log_every_n_steps=100,
                          resume_from_checkpoint=checkpoint,
                          max_epochs= cfg['train']['max_epoch'],
                          callbacks=[lr_monitor, checkpoint_saver],
                          num_sanity_val_steps=1,
                        #   profiler="advanced"
                          )


    # Train!
    if test:
        print('TESTING MODE')
        trainer.test(model, data)
    else:
        print('TRAINING MODE')
        trainer.fit(model, data)
        
class SemanticSegmenter(LightningModule):
    def __init__(self, hparams:dict, data_module: LightningDataModule = None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module
        
        
        self.model = minkunet_blocks.MinkUNet14D(in_channels=3, out_channels=2)
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([10.0, 1.0]))
        # Intersection over Union
        self.jaccard = JaccardIndex(task="multiclass", num_classes=2, ignore_index=-1, average='none')

    def points_to_tensor(self, x_feats, batch_size, resolution):
        x_feats = ME.utils.batched_coordinates(list(x_feats[:]), dtype=torch.float32, device=self.device)

        x_coord = x_feats[:,:4].clone()
        x_coord[:,1:] = torch.round(x_feats[:,1:4] / resolution)
        # x_coord[:,1:] = feats_to_coord(x_feats[:,1:4], resolution, batch_size)
        x_t = ME.TensorField(
            features=x_feats[:,1:],
            coordinates=x_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        torch.cuda.empty_cache()

        return x_t

    def training_step(self, batch:dict, batch_idx):
        torch.cuda.empty_cache()
        
        coords = batch['pcd_conditioning_pts']
        input = self.points_to_tensor(coords, len(batch['pcd_conditioning_pts']), resolution=self.hparams['data']['cond_resolution'])

        # Forward
        output = self.model(input.sparse())
        sliced_output = output.slice(input)
        labels = torch.cat(batch['scan_point_classes']).long()
        loss = self.criterion(sliced_output.F, labels)

        self.log('train/loss', loss, on_step=True)
        torch.cuda.empty_cache()

        return loss
    
  
    def validation_step(self, batch:dict, batch_idx):
        self.model.eval()
        
        # coords, feat, label = data_loader(is_classification=False)
        coords = batch['pcd_conditioning_pts']
        input = self.points_to_tensor(coords, len(batch['pcd_conditioning_pts']), resolution=self.hparams['data']['cond_resolution'])

        # Forward
        output = self.model(input.sparse())
        sliced_output = output.slice(input)
        labels = torch.cat(batch['scan_point_classes']).long()
        preds = sliced_output.F.argmax(1)
        self.jaccard(preds, labels) # update

        torch.cuda.empty_cache()
        return 
    
    def on_validation_epoch_end(self):
        ious = self.jaccard.compute()
        self.log('val/iou_0', ious[0], prog_bar=True)
        self.log('val/iou_1', ious[1])
        self.log('val/mean_iou', (ious[0] + ious[1]) / 2, prog_bar=True)
        self.jaccard.reset()
        return  {'val/iou_0': ious[0], 'val/iou_1': ious[1], 'val/mean_iou': (ious[0] + ious[1]) / 2}

    def test_step(self, batch:dict, batch_idx):
        self.model.eval()
        
        coords = batch['pcd_conditioning_pts']
        input = self.points_to_tensor(coords, len(batch['pcd_conditioning_pts']), resolution=self.hparams['data']['cond_resolution'])

        # Forward
        output = self.model(input.sparse())
        sliced_output = output.slice(input)
        labels = torch.cat(batch['scan_point_classes']).long()

        preds = sliced_output.F.argmax(1)
        self.jaccard(preds, labels) # update
        
        
        # save the predictions as npz
        os.makedirs(f'{self.logger.log_dir}/generated_pcd/', exist_ok=True)
        for batch_it in range(len(coords)):
            output_name = os.path.basename(batch['filename'][batch_it]).replace('.pt', '.npz')
            pred_batch_labels = preds[len(coords[batch_it])*batch_it:len(coords[batch_it])*(batch_it+1)].cpu().numpy()
            np.savez(f'{self.logger.log_dir}/generated_pcd/{output_name}', pred_sem=pred_batch_labels, coords=coords[batch_it].cpu().numpy())

        return {'test/iou': None}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['train']['lr'], betas=(0.9, 0.999))

        return optimizer

if __name__ == "__main__":
    main(auto_envvar_prefix='PLS')
