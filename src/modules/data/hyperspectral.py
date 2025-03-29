import os
import requests
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

from src.modules.transform import NonlinearComponentWise as NonlinearTransform
from src.modules.transform import HyperspectralTransform
from src.modules.utils import init_plot, run_dir


# fixme: add labels for pavia, add moffitt
class DataModule(LightningDataModule):
    def __init__(self, data_config, transform=None, **config):
        super().__init__()
        self.data_config = data_config
        self.batch_size = config.get("batch_size")
        self.val_batch_size = config.get("val_batch_size")
        self.num_workers = config.get("num_workers")
        self.shuffle = config.get("shuffle")
        self.dataset = None
        self.dataset_size = data_config["dataset_size"]
        self.nonlinearity = data_config.get("nonlinearity")
        self.tensors = {}
        # self.latent_dim = config["latent_dim"]
        # fixme: add latent_dim to data_config to choose latent tru

    def prepare_data(self):
        observed_data, labels = self.import_data()

        self.tensors['noiseless_sample'] = torch.tensor(observed_data, dtype=torch.float32).permute(2, 0, 1)
        self.tensors['noisy_sample'] = self.add_gaussian_noise(self.tensors['noiseless_sample'], self.data_config.get("snr"))
        # fixme: don't add noise to real data, only semi-real, figure out how to set sigma in config

        if 'latent_sample' in labels:
            self.tensors['latent_sample'] = torch.tensor(labels['latent_sample'], dtype=torch.float32) if labels is not None else None

        if 'linearly_mixed_sample' in labels:
            self.tensors['linearly_mixed_sample'] = torch.tensor(labels['linearly_mixed_sample'], dtype=torch.float32).permute(2, 0, 1) if labels['linearly_mixed_sample'] is not None else None

        if self.dataset_size is None:
            self.dataset_size = self.tensors['noiseless_data'][0].numel()

        self.transform = HyperspectralTransform(
            output_channels=self.data_config.get("observed_dim"),
            normalize=self.data_config.get("normalize", True),
            dataset_size=self.dataset_size
        )

        self.latent_transform = HyperspectralTransform(
            output_channels=self.data_config.get("latent_dim"),
            normalize=self.data_config.get("normalize", True),
            dataset_size=self.dataset_size
        )

    def import_data(self):
        dataset = self.data_config["data_model"]
        data_dir = os.path.join(os.path.abspath(__file__).split("src")[0],
                                 f'datasets/hyperspectral/{dataset.split("-")[0]}')

        if dataset == "PaviaU":
            observed_data = sio.loadmat(data_dir + "/data.mat")['paviaU']
            latent_sample = None

            labels = {
            }
            # url = "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat"
            # if not os.path.exists(file_path):
            #     with open(file_path, "wb") as f:
            #         f.write(requests.get(url).content)

        if dataset == "Urban_R162":
            mat_data = sio.loadmat(data_dir + "/data.mat")
            n_row = int(mat_data['nRow'][0, 0])
            n_col = int(mat_data['nCol'][0, 0])

            latent_sample = sio.loadmat(data_dir + "/end5_groundTruth.mat")['A']
            latent_sample = np.copy(latent_sample).reshape(-1, n_row, n_col)

            mixing = sio.loadmat(data_dir + "/end5_groundTruth.mat")['M']
            labels_flat = torch.flatten(torch.tensor(latent_sample), start_dim=1).T
            mixing = torch.tensor(mixing)

            linearly_mixed_sample = mixing @ labels_flat.T

            observed_data = mat_data['Y'].reshape(-1, n_row, n_col).transpose(1, 2, 0)

            linearly_mixed_sample = linearly_mixed_sample.reshape(-1, n_row, n_col).permute(1, 2, 0).detach().clone().numpy()
            latent_sample = latent_sample.transpose(1, 2, 0)

            labels = {
                'latent_sample': latent_sample,
                'linearly_mixed_sample': linearly_mixed_sample
            }

        if dataset == "Urban_R162-semi":
            mat_data = sio.loadmat(data_dir + "/data.mat")
            n_row = int(mat_data['nRow'][0, 0])
            n_col = int(mat_data['nCol'][0, 0])

            latent_sample = sio.loadmat(data_dir + "/end5_groundTruth.mat")['A']
            latent_sample = np.copy(latent_sample).reshape(-1, n_row, n_col)

            mixing = sio.loadmat(data_dir + "/end5_groundTruth.mat")['M']
            labels_flat = torch.flatten(torch.tensor(latent_sample), start_dim=1).T
            mixing = torch.tensor(mixing)

            linearly_mixed_sample = mixing @ labels_flat.T

            nonlinear_transform = NonlinearTransform(
                latent_dim=None,
                observed_dim=mixing.shape[0],
                degree=None,
                nonlinearity=self.nonlinearity
            ).requires_grad_(False)

            observed_data = nonlinear_transform(linearly_mixed_sample)
            observed_data = observed_data.reshape(-1, n_row, n_col).permute(1, 2, 0).detach().clone().numpy()

            linearly_mixed_sample = linearly_mixed_sample.reshape(-1, n_row, n_col).permute(1, 2, 0).detach().clone().numpy()
            latent_sample = latent_sample.transpose(1, 2, 0)

            labels = {
                'latent_sample': latent_sample,
                'linearly_mixed_sample': linearly_mixed_sample
            }

        return observed_data, labels

    def setup(self, stage=None):
        transformed_data = self.transform(self.tensors['noisy_sample']).detach().cpu()

        labels = {}

        if 'noiseless_sample' in self.tensors:
            noiseless_data = self.transform(self.tensors['noiseless_sample']).detach().cpu()
            labels['noiseless_sample'] = noiseless_data

        if 'linearly_mixed_sample' in self.tensors:
            linearly_mixed_sample = self.transform(self.tensors['linearly_mixed_sample']).detach().cpu() if self.tensors['linearly_mixed_sample'] is not None else None
            labels['linearly_mixed_sample'] = linearly_mixed_sample

        if 'latent_sample' in self.tensors:
            latent_sample = self.latent_transform(self.tensors['latent_sample'].permute(2, 0, 1)).detach().cpu() if self.tensors['latent_sample'] is not None else None
            self.tensors['latent_sample_qr'], _ = np.linalg.qr(latent_sample)
            labels['latent_sample'] = latent_sample
            labels['latent_sample_qr'] = self.tensors['latent_sample_qr']

        if self.data_config["normalize"]:
            self.sigma /= self.tensors['noiseless_sample'].max() - self.tensors['noiseless_sample'].min()

        self.dataset = HyperspectralDataset(data=transformed_data, labels=labels)

    def add_gaussian_noise(self, tensor, snr_db):
        noise_power = tensor.pow(2).mean() / (10 ** (snr_db / 10))
        self.sigma = torch.sqrt(noise_power)
        return tensor + torch.randn_like(tensor) * self.sigma

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.dataset_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.dataset_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.dataset_size, shuffle=False, num_workers=self.num_workers)

    def plot(self, layer_index=0, **kwargs):
        init_plot()
        datasets = list(kwargs.values())
        names = list(kwargs.keys())
        height, width = datasets[0].shape[1], datasets[0].shape[2]

        fig, axs = plt.subplots(1, len(datasets), figsize=(3 * len(datasets), 2.5), dpi=300)
        axs = np.atleast_1d(axs)

        for idx, dataset in enumerate(datasets):

            # if idx > 0:
            #     height, width = self.transform.calculate_transformed_dimensions(height, width)

            # layer = dataset[layer_index].reshape(height, width).detach().cpu().numpy()
            layer = dataset[layer_index].detach().cpu().numpy()

            if None:
                norm = mcolors.Normalize(vmin=np.min(layer), vmax=np.max(layer))
            else:
                norm = None
            im = axs[idx].imshow(layer, cmap='viridis', norm=norm)
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


def plot_components(labels=None, scale=False, show_plot=False, save_plot=False, name=None, max_points=10e8, **kwargs):
    import os
    plt = init_plot()
    A4_WIDTH = 8.27

    num_components = kwargs[list(kwargs.keys())[0]][0].shape[-1]

    current = num_components
    while not any(current % i == 0 for i in range(3, 7)):
        current += 1
    n_cols = next(i for i in range(3, 7) if current % i == 0)
    n_rows = (num_components + n_cols - 1) // n_cols

    aspect_ratio = 1.0
    fig_width = A4_WIDTH
    fig_height = fig_width * n_rows / n_cols * aspect_ratio

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), dpi=300)
    axes = np.atleast_1d(axes.flatten())

    markers = ['o', 'x', '^', 's', 'd']
    marker_size = 9

    for i in range(num_components):
        for j, (k, (x, v)) in enumerate(kwargs.items()):
            x_component = x[..., i].clone().detach().cpu().numpy()
            y_component = v(x)[..., i].clone().detach().cpu().numpy() if callable(v) else v[..., i].clone().detach().cpu().numpy()

            if (torch.max(torch.tensor(y_component)) - torch.min(torch.tensor(y_component))).item() < 1e-6:
                continue

            if len(x_component) > max_points:
                indices = torch.randperm(len(x_component))[:int(max_points)]
                x_component = x_component[indices]
                y_component = y_component[indices]

            marker = markers[j % len(markers)]
            axes[i].scatter(
                visual_normalization(torch.tensor(x_component)) if scale else x_component,
                visual_normalization(torch.tensor(y_component)) if scale else y_component,
                label=k.replace('_', ' ').capitalize(),
                marker=marker,
                s=marker_size
            )

        axes[i].set_title(f"Component {i + 1}")
        # axes[i].legend()
        axes[i].grid(True)

        # axes[i].set_xlabel(r"$z_{true}$", fontsize=10)
        # axes[i].set_ylabel(r"$z_{est}$", fontsize=10)

    for i in range(num_components, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_plot:
        dir = '../../../notebooks/figures/'#run_dir('predictions')
        os.makedirs(dir, exist_ok=True)
        path = f"{dir}/{name}.png"
        fig.savefig(path, transparent=True)
        print(f"Saved {name} plot to '{path}'")

    if show_plot:
        plt.show()

    plt.close()
    return plt


def visual_normalization(x):
    bound = 1
    x = x - torch.min(x)
    x = x / (torch.max(x)) * bound
    return x


if __name__ == "__main__":

    data_config = {
        'data_model': 'Urban_R162-semi',
        'nonlinearity': 'cnae',
        'scale': 5,
        'observed_dim': 16,
        'nonlinear_transform_init': None,
        'mixing_matrix_init': None,
        'normalize': True,
        'degree': None,
        'snr': 25,
        'dataset_size': 1000,
        'data_seed': 12
    }

    config = {
        'module_name': 'synthetic',
        'split': [1.0, 0.0, 0.0],
        'batch_size': 100,
        'val_batch_size': 10000,
        'num_workers': 4,
        'shuffle': True
    }

    # fixme: refactor data_model and nonlinearity

    data_module = DataModule(
        data_config,
        transform=HyperspectralTransform(
            normalize=True,
            output_channels=4,
            dataset_size=data_config['dataset_size']),
        **config
    )
    data_module.prepare_data()
    data_module.setup()

    original_data = data_module.tensors['noiseless_data']
    processed_data = data_module.transform.unflatten(data_module.transform(original_data))

    data_module.plot(layer_index=0, original_data=original_data, processed_data=processed_data)

    data_config = {
        'data_model': 'Urban_R162-semi',
        'nonlinearity': None,
        'scale': 5,
        'observed_dim': 16,
        'nonlinear_transform_init': None,
        'mixing_matrix_init': None,
        'normalize': True,
        'degree': None,
        'snr': 25,
        'dataset_size': 1000,
        'data_seed': 12
    }

    config = {
        'module_name': 'synthetic',
        'split': [1.0, 0.0, 0.0],
        'batch_size': 100,
        'val_batch_size': 10000,
        'num_workers': 4,
        'shuffle': True
    }

    data_module = DataModule(
        data_config,
        transform=HyperspectralTransform(
            normalize=True,
            output_channels=4,
            dataset_size=data_config['dataset_size']),
        **config
    )
    data_module.prepare_data()
    data_module.setup()

    original_data_real = data_module.tensors['noiseless_data']
    processed_data_real = data_module.transform.unflatten(data_module.transform(original_data_real))

    # data_module.plot(layer_index=0, original_data=original_data_real, processed_data=processed_data_real)

    data_module.plot(layer_index=0, semi_real_data=original_data, real_data=original_data_real)

    plot_components(
        nonlinearity=(torch.flatten(processed_data, start_dim=1).T, torch.flatten(processed_data_real, start_dim=1).T),
        scale=True,
        show_plot=True,
        save_plot=True,
        name="true_vs_linearly_mixed"
    )

# todo: separate val dataloader and test, for synthetic and hyperspectral
