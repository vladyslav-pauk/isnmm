import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.seed import isolate_rng

import src.modules.distribution as probability_model


class DataModule(pl.LightningDataModule):
    def __init__(self, model_config, observed_dim, latent_dim, size, batch_size, split, num_workers, seed):
        super().__init__()

        self.config_data_model = model_config
        self.observed_dim = observed_dim
        self.latent_dim = latent_dim
        self.size = size
        self.batch_size = batch_size
        self.split_sizes = split
        self.num_workers = num_workers
        self.seed = seed

        self.dataset = None
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.data_model = None

    # def __init__(self, config_data_model, observed_dim=None, latent_dim=None, size=None, batch_size=None, split=None):
    #     super().__init__()
    #
    #     mixture_matrix = torch.randn(observed_dim, latent_dim)
    #     data_model_class = getattr(data_model_package, config_data_model["model_name"])
    #
    #     self.data_model = data_model_class(mixture_matrix, **config_data_model)
    #     self.lin_transform = self.data_model.linear_mixing
    #     self.nonlinear_transform = self.data_model.nonlinear_transform
    #
    #     self.config_data_model = config_data_model
    #
    #     self.observed_dim = observed_dim
    #     self.latent_dim = latent_dim
    #     self.batch_size = batch_size
    #     self.split_sizes = split
    #     self.size = size
    #
    #     self.prepare_data()
    # def prepare_data(self) -> None:
    #     self.dataset = SyntheticDataset(self.data_model, self.size)
    #     self.sigma = self.dataset.sigma

    def setup(self, stage: str = None):
        if self.seed:
            with isolate_rng():
                seed_everything(self.seed, workers=True)

                if not self.dataset:
                    # todo: loading mixture matrix from file
                    mixture_matrix = torch.randn(self.observed_dim, self.latent_dim)
                    data_model_class = getattr(probability_model, self.config_data_model["model_name"])
                    self.data_model = data_model_class(mixture_matrix, **self.config_data_model)

                    self.dataset = SyntheticDataset(self.data_model, self.size)

                    self.data_train, self.data_val, self.data_test = random_split(
                        self.dataset, self.split_sizes
                    )

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)


class SyntheticDataset(Dataset):
    def __init__(self, data_model, size):
        self.data_size = size
        self.data, self.labels = data_model.sample(sample_shape=torch.Size([size]))

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return self.data[idx], tuple(l[idx] for l in self.labels)

    # todo: generate data on-the-fly in the __getitem__ method to save memory
    #  (in this case make sure splitting is handled, random_split won't work - "9. Handle Data Splitting Appropriately")
    # def __getitem__(self, idx):
    #     # Optionally, use the idx to seed the random generator for reproducibility
    #     # torch.manual_seed(idx)
    #     data, labels = self.data_model.sample(sample_shape=torch.Size([1]))
    #     data = data.squeeze(0)
    #     labels = tuple(label.squeeze(0) for label in labels)
    #     return data, labels
