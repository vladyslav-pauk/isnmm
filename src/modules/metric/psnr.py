import torch
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt

from src.modules.data.hyperspectral import DataModule
from src.modules.transform.convolution import HyperspectralTransform
from src.utils.wandb_tools import run_dir


class PSNR(torchmetrics.Metric):
    def __init__(self, max_val=255, dist_sync_on_step=False, show_plot=False, log_plot=False, save_plot=False, image_dims=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.max_val = max_val
        self.show_plot = show_plot
        self.save_plot = save_plot
        self.image_dims = image_dims
        self.add_state("psnr_values", default=[], dist_reduce_fx="mean")

    def update(self, reconstructed, target):
        mse = ((reconstructed - target) ** 2)
        psnr = 10 * torch.log10(self.max_val ** 2 / mse)
        self.psnr_values.append(psnr)

    def compute(self):
        psnr_avg = torch.mean(torch.stack(self.psnr_values))
        self.plot_data({"psnr": self.psnr_values})
        self.psnr_values.clear()
        return psnr_avg

    def plot_data(self, plot_data):
        channels, height, width = self.image_dims

        num_components = next(iter(plot_data.values()))[0].shape[-1]

        rows = (num_components + 2) // 3
        cols = 3

        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(3 * cols, 4.5 * rows), dpi=300)
        axs = np.atleast_2d(axs)  # Ensure axs is at least 2D

        for comp_idx in range(num_components):

            for idx, (key, data) in enumerate(plot_data.items()):

                data = data[0].view(num_components, height, width)
                component = data[comp_idx].cpu().numpy()

                row = comp_idx // cols
                col = comp_idx % cols

                axs[row, col].imshow(component, cmap='viridis')  # Show first channel
                axs[row, col].set_title(f'{key.replace("_", " ").capitalize()}, {comp_idx}')
                axs[row, col].axis('off')

        plt.tight_layout()

        if self.show_plot:
            plt.show()
        if self.save_plot:
            dir = run_dir('predictions')
            plt.savefig(f"{dir}/component_{comp_idx}.png", transparent=True, dpi=300)
            print(f"Saved PSNR for component {comp_idx} image to '{dir}/component_{comp_idx}.png'")

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

    # Initialize the data module and transform
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

    # Initialize the PSNR metric and compute
    psnr_metric = PSNR(show_plot=True, save_plot=False)
    psnr_metric.update(reconstructed_images, observed_images)
    psnr_value = psnr_metric.compute()

    print(f"PSNR: {psnr_value}")