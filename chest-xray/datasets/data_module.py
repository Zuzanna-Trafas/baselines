import pytorch_lightning as pl

from torch.utils.data import DataLoader
from . import image_dataset
import builder


class MIMICDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.dataset = image_dataset.MIMICImageDataset

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "valid")
        dataset = self.dataset(self.cfg, split="validate", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )