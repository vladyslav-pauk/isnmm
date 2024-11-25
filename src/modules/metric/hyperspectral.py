import numpy as np

import torch
import torchmetrics

from src.modules.data.hyperspectral import DataModule
from src.modules.transform.convolution import HyperspectralTransform
from src.utils.wandb_tools import run_dir
from src.utils.utils import init_plot
# todo: rewrite using torch add_state interface


class Hyperspectral(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, show_plot=False, log_plot=True, save_plot=True, image_dims=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.show_plot = show_plot
        self.log_plot = log_plot
        self.save_plot = save_plot
        self.image_dims = image_dims

        self.state_data = {}

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.state_data:
                self.state_data[key] = []
            self.state_data[key].append(value.clone().detach().cpu())

    def compute(self):
        plot_data = {key: torch.cat(val, dim=0) for key, val in self.state_data.items() if key != 'labels'}
        self.plot_data(plot_data)
        self.state_data.clear()
        return {}

    # def plot_data(self, plot_data):
    #     channels, height, width = self.image_dims
    #     for key, data in plot_data.items():
    #     data = data.view(channels, height, width)
    #     for i in range(data.shape[0]):
    #         fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    #         ax.imshow(data[i].cpu().numpy(), cmap='viridis')
    #         ax.set_title(f'{key.replace('_', ' ').capitalize()}, {i} component')
    #         ax.axis('off')
    #
    #         if self.show_plot:
    #             plt.show()
    #         if self.save_plot:
    #             dir = run_dir('predictions')
    #             plt.savefig(f"{dir}/{key}_component_{i}.png", transparent=True, dpi=300)
    #             print(f"Saved {key} component {i} image to '{dir}{key}_component_{i}.png'")
    #
    #         plt.close()

    def plot_data(self, plot_data):
        _, height, width = self.image_dims

        plt = init_plot()

        key, data = next(iter(plot_data.items()))

        data = data.T.view(-1, height, width)

        num_components = data.shape[0]

        rows = (num_components + 2) // 3

        if len(plot_data) == 1:
            fig, axs = plt.subplots(rows, 3, figsize=(9, 4.5 * rows), dpi=300)
            axs = np.atleast_2d(axs)

            for i in range(num_components):
                row = i // 3
                col = i % 3
                component = data[i].cpu().numpy()
                axs[row, col].imshow(component, cmap='viridis')
                axs[row, col].set_title(f'{key.replace("_", ' ').capitalize()} {i+1}')
                axs[row, col].axis('off')

            plt.tight_layout()
            if self.show_plot:
                plt.show()
            if self.save_plot:
                dir = run_dir('predictions')
                plt.savefig(f"{dir}/{key}-components.png", transparent=True, dpi=300)
                print(
                    f"Saved {key} components image to '{dir}/{key}_components.png'")
            plt.close()

        else:
            for comp_idx in range(num_components):
                fig, axs = plt.subplots(1, len(plot_data), figsize=(3 * len(plot_data), 4.5), dpi=300)
                axs = np.atleast_1d(axs)

                for idx, (key, data) in enumerate(plot_data.items()):
                    data = data.T.view(-1, height, width)
                    component = data[comp_idx].cpu().numpy()

                    axs[idx].imshow(component, cmap='viridis')
                    axs[idx].set_title(f'{key.replace("_", " ").capitalize()} {comp_idx+1}')
                    axs[idx].axis('off')

                plt.tight_layout()

                if self.show_plot:
                    plt.show()
                if self.save_plot:
                    dir = run_dir('predictions')
                    plt.savefig(f"{dir}/component_{comp_idx}.png", transparent=True, dpi=300)
                    print(
                        f"Saved {', '.join(list(plot_data.keys()))} component {comp_idx} image to '{dir}/{key}_component_{comp_idx}.png'")

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
        "dataset_size": 1000,
        "observed_dim": 3,
        "latent_dim": 3
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
    transformed_images = transformed_data

    dat = data_module.transform.inverse(transformed_data)
    reconstructed_images = data_module.transform.flatten(dat)

    img_dims = (data_config["observed_dim"], data_module.transform.height, data_module.transform.width)

    hyperspectral_metric = Hyperspectral(show_plot=True, save_plot=False, image_dims=img_dims)
    hyperspectral_metric.update(recovered_abundances=reconstructed_images, transformed_images=transformed_images)
    hyperspectral_metric.compute()
