import numpy as np
import matplotlib.pyplot as plt

import torch
import torchmetrics

from src.modules.data.hyperspectral import DataModule
from src.modules.transform.convolution import HyperspectralTransform
from src.utils.wandb_tools import run_dir
from src.utils.utils import init_plot
# todo: rewrite using torch add_state interface


class Hyperspectral(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, show_plot=False, log_plot=True, save_plot=True, image_dims=None, unmixing=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.show_plot = show_plot
        self.log_plot = log_plot
        self.save_plot = save_plot
        self.image_dims = image_dims

        self.unmixing = unmixing
        self.state_data = {}

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.state_data:
                self.state_data[key] = []
            self.state_data[key].append(value.clone().detach().cpu())

    def compute(self):
        for key, value in self.state_data.items():
            self.state_data[key] = torch.cat(value, dim=0)
        # plot_data = {key: torch.cat(val, dim=0) for key, val in self.state_data.items() if key != 'labels'}
        if self.unmixing:
            self.state_data["abundance"], mixing_matrix = self.unmix(self.state_data["abundance"], latent_dim=self.image_dims[0], model=self.unmixing)
            mixing_matrix_pinv = torch.linalg.pinv(mixing_matrix)

            self.state_data["noise"] = torch.matmul(mixing_matrix_pinv, self.state_data["noise"].T).T

        plot_data = {key: val for key, val in self.state_data.items() if key != 'labels'}
        self.plot_data(plot_data)
        self.state_data.clear()
        return {}

    def unmix(self, latent_sample, latent_dim, model=None):
        import src.model as model_package

        dataset_size = latent_sample.size(0)
        unmixing_model = getattr(model_package, model).Model
        unmixing = unmixing_model(
            latent_dim=latent_dim,
            dataset_size=dataset_size
        )
        latent_sample, mixing_matrix = unmixing.estimate_abundances(latent_sample.squeeze().cpu().detach())
        # unmixing.plot_multiple_abundances(latent_sample, [0,1,2,3,4,5,6,7,8,9])
        # unmixing.plot_mse_image(rows=100, cols=10)

        return latent_sample, mixing_matrix

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
        plt = init_plot()

        _, height, width = self.image_dims

        all_data = torch.cat([data.T.view(-1, height, width) for data in plot_data.values()], dim=0)
        global_min = all_data.min().item()
        global_max = all_data.max().item()
        print(f"Global normalization: min={global_min}, max={global_max}")

        for key, data in plot_data.items():
            data = data.T.view(-1, height, width)
            num_components = data.shape[0]

            cols = 4
            rows = (num_components + 2) // cols

            fig, axs = plt.subplots(rows, cols, figsize=(9, 4.5 * rows), dpi=300)
            axs = np.atleast_2d(axs)

            for i in range(num_components):
                row, col = divmod(i, cols)
                component = data[i].cpu().numpy()
                axs[row, col].imshow(component, cmap='viridis', vmin=global_min, vmax=global_max)
                axs[row, col].set_title(f'{key.replace("_", " ").capitalize()} {i + 1}')
                axs[row, col].axis('off')

            for i in range(num_components, rows * cols):
                row, col = divmod(i, cols)
                axs[row, col].axis('off')

            plt.tight_layout()

            if self.save_plot:
                dir = run_dir('predictions')
                plt.savefig(f"{dir}/{key}_components.png", transparent=True, dpi=300)
                print(f"Saved {key} components image to '{dir}/{key}_components.png'")

            if self.show_plot:
                plt.show()

            plt.close()

    # def plot_data(self, plot_data):
    #     _, height, width = self.image_dims
    #
    #     plt = init_plot()
    #
    #     key, data = next(iter(plot_data.items()))
    #
    #     data = data.T.view(-1, height, width)
    #
    #     num_components = data.shape[0]
    #
    #     rows = (num_components + 2) // 3
    #
    #     if len(plot_data) == 1:
    #         fig, axs = plt.subplots(rows, 3, figsize=(9, 4.5 * rows), dpi=300)
    #         axs = np.atleast_2d(axs)
    #
    #         for i in range(num_components):
    #             row = i // 3
    #             col = i % 3
    #             component = data[i].cpu().numpy()
    #             axs[row, col].imshow(component, cmap='viridis')
    #             axs[row, col].set_title(f'{key.replace("_", ' ').capitalize()} {i+1}')
    #             axs[row, col].axis('off')
    #
    #         plt.tight_layout()
    #         if self.show_plot:
    #             plt.show()
    #         if self.save_plot:
    #             dir = run_dir('predictions')
    #             plt.savefig(f"{dir}/{key}-components.png", transparent=True, dpi=300)
    #             print(
    #                 f"Saved {key} components image to '{dir}/{key}_components.png'")
    #         plt.close()
    #
    #     else:
    #         for comp_idx in range(num_components):
    #             fig, axs = plt.subplots(1, len(plot_data), figsize=(3 * len(plot_data), 4.5), dpi=300)
    #             axs = np.atleast_1d(axs)
    #
    #             for idx, (key, data) in enumerate(plot_data.items()):
    #                 data = data.T.view(-1, height, width)
    #                 component = data[comp_idx].cpu().numpy()
    #                 axs[idx].imshow(component, cmap='viridis')
    #                 axs[idx].set_title(f'{key.replace("_", " ").capitalize()} {comp_idx+1}')
    #                 axs[idx].axis('off')
    #
    #             plt.tight_layout()
    #
    #             if self.show_plot:
    #                 plt.show()
    #             if self.save_plot:
    #                 dir = run_dir('predictions')
    #                 plt.savefig(f"{dir}/component_{comp_idx}.png", transparent=True, dpi=300)
    #                 print(
    #                     f"Saved {', '.join(list(plot_data.keys()))} component {comp_idx} image to '{dir}/{key}_component_{comp_idx}.png'")
    #
    #             plt.close()


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
