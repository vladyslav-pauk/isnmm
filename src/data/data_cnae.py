import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
import scipy.io as sio


class MyDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor
        self.data_len = data_tensor.shape[0]

    def __getitem__(self, index):
        return self.data[index], index

    def __len__(self):
        return self.data_len


class DataModule(LightningDataModule):
    def __init__(self, model_config=None, observed_dim=None, latent_dim=None, num_samples=None, batch_size=None, split=None):
        super().__init__()
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.observed_dim = observed_dim
        self.dataset_size = num_samples
        self.linear_mixture = None

    def prepare_data(self):
        data_file = 'datasets/pnl/post-nonlinear_simplex_synthetic_data.mat'
        data = sio.loadmat(data_file)
        self.x = torch.tensor(data['x'], dtype=torch.float32)
        self.linear_mixture = torch.tensor(data['linear_mixture'], dtype=torch.float32)
        # todo: might use self.data and call whatever needed from the model

    def setup(self, stage=None):
        self.dataset = MyDataset(self.x)
        self.n_feature = self.x.shape[1]

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)