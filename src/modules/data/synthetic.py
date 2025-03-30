import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
import scipy.io as sio

from ..utils import dict_to_str


class DataModule(LightningDataModule):
    def __init__(self, data_config, **config):
        super().__init__()
        self.data_config = data_config
        # self.model_name = data_config["data_module_name"]
        self.dataset_size = data_config["dataset_size"]
        self.latent_dim = data_config["latent_dim"]
        self.observed_dim = data_config["observed_dim"]
        self.snr_db = data_config["snr"]
        self.seed = data_config["data_seed"]
        self.sigma = None

        self.batch_size = config["batch_size"]
        self.val_batch_size = config["val_batch_size"]
        self.split = config["split"]
        self.num_workers = config["num_workers"]
        self.shuffle = config['shuffle']

        self.linear_mixture = None

        self.observed_sample = None
        self.latent_sample = None
        self.linearly_mixed_sample = None

    def prepare_data(self):
        dataset_name = dict_to_str(self.data_config)
        data_file = f'datasets/synthetic/{dataset_name}.mat'

        data = sio.loadmat(data_file)

        self.latent_dim = data['latent_sample'][0, 0]
        self.observed_dim = data['observed_sample'][0, 0]

        self.observed_sample = torch.tensor(data['observed_sample'][:self.dataset_size], dtype=torch.float32)
        self.latent_sample = torch.tensor(data['latent_sample'][:self.dataset_size], dtype=torch.float32)
        self.noiseless_sample = torch.tensor(data['noiseless_sample'][:self.dataset_size], dtype=torch.float32)
        self.linearly_mixed_sample = torch.tensor(data['linearly_mixed_sample'][:self.dataset_size], dtype=torch.float32)
        self.latent_data_qr = torch.tensor(data['latent_sample_qr'][:self.dataset_size], dtype=torch.float32)
        self.linear_mixture = torch.tensor(data['linear_mixture'], dtype=torch.float32)
        self.sigma = data['sigma'][0, 0]

    def setup(self, stage=None):
        self.dataset = MyDataset(
            data=self.observed_sample,
            labels={
                "latent_sample": self.latent_sample,
                "linearly_mixed_sample": self.linearly_mixed_sample,
                "noiseless_sample": self.noiseless_sample,
                "latent_sample_qr": self.latent_data_qr
            })
        self.n_feature = self.observed_sample.shape[1]

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.dataset_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.dataset_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.dataset_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.data_size = self.data.size(0)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        indexed_labels = {key: value[idx] for key, value in self.labels.items()}
        return {
            "data": self.data[idx],
            "labels": indexed_labels,
            "idxes": idx
        }
