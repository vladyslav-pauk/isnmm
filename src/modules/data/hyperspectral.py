import os
import requests
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

from src.modules.transform.convolution import HyperspectralTransform


class DataModule(LightningDataModule):
    def __init__(self, data_config, transform=None, **config):
        super().__init__()
        self.data_config = data_config
        self.batch_size = config.get("batch_size")
        self.val_batch_size = config.get("val_batch_size")
        self.num_workers = config.get("num_workers")
        self.shuffle = config.get("shuffle")
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
        noiseless_data = self.transform(self.tensor_data).detach().cpu()

        labels = {
            "noiseless_data": noiseless_data
        }
        self.dataset = HyperspectralDataset(data=transformed_data, labels=labels)

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
        return DataLoader(self.dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers)

    def plot(self, layer_index=0, **kwargs):
        datasets = list(kwargs.values())
        names = list(kwargs.keys())
        height, width = datasets[0].shape[1], datasets[0].shape[2]

        fig, axs = plt.subplots(1, len(datasets), figsize=(3 * len(datasets), 4.2), dpi=300)
        axs = np.atleast_1d(axs)

        for idx, dataset in enumerate(datasets):

            if idx > 0:
                height, width = self.transform.calculate_transformed_dimensions(height, width)

            layer = dataset[layer_index].reshape(height, width).detach().cpu().numpy()

            im = axs[idx].imshow(layer, cmap='viridis', norm=mcolors.Normalize(vmin=np.min(layer), vmax=np.max(layer)))
            axs[idx].set_title(f'{names[idx].replace("_", " ").capitalize()}, 0 layer')
            axs[idx].axis('off')

        fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.58, pad=0.0)
        fig.subplots_adjust(left=-0.1, right=0.94, top=0.89, bottom=0.1)
        path = f"../../../notebooks/figures/hyperspectral_data_processing.png"
        plt.savefig(path, transparent=True, dpi=300)
        print(f"Saved hyperspectral data processing plot to '{path}'")
        plt.show()


class HyperspectralDataset(Dataset):
    def __init__(self, data, labels=None):
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


if __name__ == "__main__":

    data_config = {
        "batch_size": 128,
        "val_batch_size": 128,
        "shuffle": True,
        "snr": 25,
        "dataset_size": 10000
    }
    config = {"num_workers": 4}

    data_module = DataModule(
        data_config,
        transform=HyperspectralTransform(
            normalize=True,
            output_channels=2,
            dataset_size=data_config['dataset_size']),
        **config
    )
    data_module.prepare_data()
    data_module.setup()

    original_data = data_module.noisy_data
    processed_data = data_module.transform.unflatten(data_module.transform(original_data))

    data_module.plot(layer_index=0, original_data=original_data, processed_data=processed_data)

    # print(f"Dataset size: {len(data_module.dataset)}")
    # for batch in data_module.train_dataloader():
    #     print(f"Batch shape: {batch['data'].shape}")
    #     break

# todo: separate val dataloader and test, for synthetic and hyperspectral
