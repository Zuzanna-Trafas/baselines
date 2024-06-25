import pytorch_lightning as pl

from torch.utils.data import DataLoader
from mimic_dataset import MIMICDataset


class MIMICDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.dataset = MIMICDataset

    def train_dataloader(self):
        dataset = self.dataset(split="train", rgb=self.cfg.data.rgb)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def val_dataloader(self):
        dataset = self.dataset(split="validate", rgb=self.cfg.data.rgb)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        dataset = self.dataset(self.cfg, split="test", rgb=self.cfg.data.rgb)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )