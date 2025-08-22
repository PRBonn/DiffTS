import warnings

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from DiffTS.datasets.dataloader.OrchardDataset import OrchardDataset
from DiffTS.datasets.dataloader.SyntheticTrees import \
    SyntheticTreesDataset
from DiffTS.datasets.dataloader.TreeNet3D import TreeNet3D
from DiffTS.utils.collations import SkeletonCollation

warnings.filterwarnings('ignore')

__all__ = ['SyntheticTreesDataModule']

class SyntheticTreesDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        collate = SkeletonCollation()

        data_set = SyntheticTreesDataset(
            data_dir=self.cfg['data']['data_dir'],
            split='train',
            cfg=self.cfg['data'],)
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate, drop_last=False)
        return loader

    def val_dataloader(self):
        collate = SkeletonCollation()

        data_set = SyntheticTreesDataset(
            data_dir=self.cfg['data']['data_dir'],
            split='val',
            cfg=self.cfg['data'],)
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=False,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate, drop_last=False)
        return loader

    def test_dataloader(self):
        collate = SkeletonCollation()
        
        data_set = SyntheticTreesDataset(
            data_dir=self.cfg['data']['data_dir'],
            split='test',
            cfg=self.cfg['data'],)
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=False,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate,)
        return loader
    
class TreeNet3DDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        collate = SkeletonCollation()

        data_set = TreeNet3D(
            data_dir=self.cfg['data']['data_dir'],
            split='train',
            cfg=self.cfg['data'],)
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate, drop_last=False)
        return loader

    def val_dataloader(self):
        collate = SkeletonCollation()

        data_set = TreeNet3D(
            data_dir=self.cfg['data']['data_dir'],
            split='val' if not self.cfg['data']['overfit'] else 'train',
            cfg=self.cfg['data'],)
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=False,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate, drop_last=False)
        return loader

    def test_dataloader(self):
        collate = SkeletonCollation()
        
        data_set = TreeNet3D(
            data_dir=self.cfg['data']['data_dir'],
            split="test",
            cfg=self.cfg['data'],)
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=False,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate,)
        return loader

class OrchardDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        collate = SkeletonCollation()

        data_set = OrchardDataset(
            data_dir=self.cfg['data']['data_dir'],
            split='train',
            cfg=self.cfg['data'],)
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate, drop_last=False)
        return loader

    def val_dataloader(self):
        collate = SkeletonCollation()

        data_set = OrchardDataset(
            data_dir=self.cfg['data']['data_dir'],
            split='val' if not self.cfg['data']['overfit'] else 'train',
            cfg=self.cfg['data'],)
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=False,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate, drop_last=False)
        return loader
    
    def test_dataloader(self):
        collate = SkeletonCollation()
        
        data_set = OrchardDataset(
            data_dir=self.cfg['data']['data_dir'],
            split="test",
            cfg=self.cfg['data'],)
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=False,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate,)
        return loader
    
dataloaders = {
    'SyntheticTrees': SyntheticTreesDataModule,
    'TreeNet3D': TreeNet3DDataModule,
    'OrchardDataset': OrchardDataModule,
}

