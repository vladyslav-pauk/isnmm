import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from torchvision import transforms
import torchvision.datasets


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", dataset_name: str = "MNIST"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.dataset = getattr(torchvision.datasets, dataset_name)

    def prepare_data(self):
        self.dataset(self.data_dir, train=True, download=True, transform=self.transform)
        self.dataset(self.data_dir, train=False, download=True, transform=self.transform)

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            mnist_full = self.dataset(self.data_dir, train=True, transform=self.transform)

            self.data_train, self.data_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        if stage == "test":
            self.data_test = self.dataset(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=0, persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=10, persistent_workers=False)

    def test_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=10, persistent_workers=False)
