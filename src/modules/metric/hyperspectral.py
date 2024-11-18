import torch
import torchmetrics
import matplotlib.pyplot as plt

from src.modules.data.hyperspectral import DataModule
from src.modules.transform.convolution import HyperspectralTransform
from src.utils.wandb_tools import run_dir


class Hyperspectral(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, show_plot=False, log_plot=True, save_plot=True, image_dims=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.show_plot = show_plot
        self.log_plot = log_plot
        self.save_plot = save_plot
        self.image_dims = image_dims

        self.state_dict = {}

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.state_dict:
                self.state_dict[key] = []
            self.state_dict[key].append(value.clone().detach().cpu())

    def compute(self):
        plot_data = {key: torch.cat(val, dim=0) for key, val in self.state_dict.items() if key != 'labels'}
        for key, data in plot_data.items():
            self.plot_data(key, data)

        self.state_dict.clear()
        return {}

    def plot_data(self, key, data):
        channels, height, width = self.image_dims
        data = data.view(channels, height, width)

        for i in range(data.shape[0]):
            fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
            ax.imshow(data[i].cpu().numpy(), cmap='viridis')
            ax.set_title(f'{key.replace('_', ' ').capitalize()}, {i} component')
            ax.axis('off')

            if self.show_plot:
                plt.show()
            if self.save_plot:
                dir = run_dir('predictions')
                plt.savefig(f"{dir}/{key}_component_{i}.png", transparent=True, dpi=300)

            plt.close()


if __name__ == "__main__":
    config = {
        "batch_size": 16,
        "val_batch_size": 16,
        "num_workers": 4,
        "shuffle": True
    }
    data_config = {
        "snr": 25,
        "dataset_size": 100,
        "observed_dim": 3,
        "latent_dim": 2
    }

    data_module = DataModule(data_config, transform=HyperspectralTransform(
        normalize=True,
        output_channels=data_config['observed_dim'],
        dataset_size=data_config["dataset_size"]
    ), **config)

    data_module.prepare_data()
    data_module.setup()

    observed_images = data_module.noisy_data
    transformed_data = data_module.transform(observed_images)
    transformed_images = data_module.transform.unflatten(transformed_data)
    reconstructed_images = data_module.transform.inverse(transformed_data)

    hyperspectral_metric = Hyperspectral(show_plot=True, save_plot=False)
    hyperspectral_metric.update(recovered_abundances=reconstructed_images, transformed_images=transformed_images)
    hyperspectral_metric.compute()
