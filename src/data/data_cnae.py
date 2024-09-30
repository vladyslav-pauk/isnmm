import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
import scipy.io as sio


class MyDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor
        self.data_len = data_tensor.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.data[index], index

    def __len__(self):
        return self.data_len


class DataModule(LightningDataModule):
    def __init__(self, data_config=None, batch_size=None, split=None, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.latent_dim = data_config["latent_dim"]
        self.observed_dim = data_config["observed_dim"]
        self.dataset_size = data_config["dataset_size"]
        self.linearly_mixed_sample = None

    def prepare_data(self):
        data_file = '../datasets/synthetic/post-nonlinear_simplex_synthetic_data.mat'
        data = sio.loadmat(data_file)
        self.observed_sample = torch.tensor(data['observed_sample'], dtype=torch.float32)
        self.linearly_mixed_sample = torch.tensor(data['linearly_mixed_sample'], dtype=torch.float32)
        # todo: might use self.data and call whatever needed from the model

    def setup(self, stage=None):
        self.dataset = MyDataset(self.observed_sample, )
        self.observed_dim = self.observed_sample.shape[1]

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    # todo: move data to modules