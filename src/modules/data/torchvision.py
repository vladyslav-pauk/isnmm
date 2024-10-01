import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import torchvision.datasets
from torch.utils.data import Dataset

class FlattenTransform:
    def __call__(self, x):
        return x.view(-1)


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return {"data": x, "labels": y, "idxes": idx}


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_config=None, batch_size=None, size=None, split=None, num_workers=None, **kwargs):
        super().__init__()
        self.dataset_name = dataset_config["dataset_name"]
        self.data_dir = dataset_config["data_dir"]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            FlattenTransform()
        ])

        self.dataset = getattr(torchvision.datasets, self.dataset_name.upper())
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.observed_dim = 784
        self.latent_dim = 200

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        self.dataset(self.data_dir, train=True, download=True, transform=self.transform)
        self.dataset(self.data_dir, train=False, download=True, transform=self.transform)

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            dataset_full = self.dataset(self.data_dir, train=True, transform=self.transform)
            dataset_full = IndexedDataset(dataset_full)

            self.data_train, self.data_val = random_split(
                dataset_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        if stage == "test":
            dataset = self.dataset(self.data_dir, train=False, transform=self.transform)
            self.data_test = IndexedDataset(dataset)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)