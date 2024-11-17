import os
import requests
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
import scipy.io as sio
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

from src.modules.transform.convolution import HyperspectralTransform


class DataModule(LightningDataModule):
    def __init__(self, data_config, transform=None, **config):
        super().__init__()
        self.data_config = data_config
        self.batch_size = config.get("batch_size", 32)
        self.val_batch_size = config.get("val_batch_size", 32)
        self.num_workers = config.get("num_workers", 0)
        self.shuffle = config.get("shuffle", True)
        self.dataset = None

        self.transform = HyperspectralTransform(
            output_channels=data_config.get("observed_dim"),
            normalize=data_config.get("normalize", True),
            dataset_size=data_config.get("dataset_size")
        )

    def prepare_data(self):
        file_path = os.path.join(os.path.abspath(__file__).split("src")[0], 'datasets/hyperspectral/PaviaU.mat')
        url = "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat"
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(requests.get(url).content)
        hyperspectral_data = sio.loadmat(file_path)['paviaU']
        self.tensor_data = torch.tensor(hyperspectral_data, dtype=torch.float32).permute(2, 0, 1)
        self.noisy_data = self.add_gaussian_noise(self.tensor_data, self.data_config.get("snr"))

    def setup(self, stage=None):
        transformed_data = self.transform(self.noisy_data).detach().cpu()
        self.observed_sample = transformed_data
        self.dataset = HyperspectralDataset(data=self.observed_sample)

    def add_gaussian_noise(self, tensor, snr_db):
        noise_power = torch.mean(tensor ** 2) / (10 ** (snr_db / 10))
        return tensor + torch.randn_like(tensor) * torch.sqrt(noise_power)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=False, num_workers=self.num_workers)

    def plot_side_by_side(self, transformed_data, layer_index):
        height, width = self.tensor_data.shape[1], self.tensor_data.shape[2]
        original_layer = self.tensor_data[layer_index].detach().cpu().numpy()
        transformed_height, transformed_width = self.transform.calculate_transformed_dimensions(height, width)
        transformed_layer = transformed_data[layer_index].detach().cpu().numpy().reshape(transformed_height,
                                                                                         transformed_width)

        expanded_layer = np.repeat(np.repeat(transformed_layer, height // transformed_height, axis=0),
                                   width // transformed_width, axis=1)

        fig, axs = plt.subplots(1, 2, figsize=(6, 4.2), dpi=300)
        axs[0].imshow(original_layer, cmap='viridis')
        axs[0].set_title(f'Original Layer {layer_index}')
        axs[0].axis('off')

        im = axs[1].imshow(expanded_layer, cmap='viridis',
                           norm=mcolors.Normalize(vmin=np.min(expanded_layer), vmax=np.max(expanded_layer)))
        axs[1].set_title(f'Transformed Layer {layer_index}')
        axs[1].axis('off')
        fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.58, pad=0.0)
        fig.subplots_adjust(left=-0.1, right=0.94, top=0.89, bottom=0.1)
        plt.show()


class HyperspectralDataset(Dataset):
    def __init__(self, data):
        self.data = data.T
        self.data_size = self.data.size(0)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return {"data": self.data[idx].detach(), "idxes": idx}


if __name__ == "__main__":
    data_config = {"batch_size": 64, "val_batch_size": 64, "num_workers": 4, "shuffle": True, "snr": 20, "dataset_size": 300}
    data_module = DataModule(data_config, transform=HyperspectralTransform(normalize=True))
    data_module.prepare_data()
    data_module.setup()
    print(f"Dataset size: {len(data_module.dataset)}")
    for batch in data_module.train_dataloader():
        print(f"Batch shape: {batch['data'].shape}")
        break
    transformed_data = data_module.transform(data_module.noisy_data)

    data_module.plot_side_by_side(transformed_data, layer_index=0)

# fixme: swape dims
# fixme: plot abundances image for each material and mix: function inverse vector to image
# fixme: observable dimension -> number out channels
