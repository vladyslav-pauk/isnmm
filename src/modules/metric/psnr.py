import torch
import torchmetrics

from src.modules.data.hyperspectral import DataModule
from src.modules.transform.convolution import HyperspectralTransform


class PSNR(torchmetrics.Metric):
    def __init__(self, max_val=255, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.max_val = max_val

        self.add_state("psnr_values", default=[], dist_reduce_fx="cat")
        self.tensors = {}

    def update(self, estimated, true):
        mse = ((estimated - true) ** 2) + 1e-12
        psnr = 10 * torch.log10(self.max_val ** 2 / mse)
        self.psnr_values.append(psnr)

    def compute(self):
        psnr_values = torch.cat(self.psnr_values, dim=0)
        psnr_avg = psnr_values.mean()

        self.psnr_values.clear()
        self.tensors = {"psnr": psnr_values}
        return psnr_avg

    def plot(self, image_dims, show_plot=False, save_plot=False):
        pass


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

    psnr_metric = PSNR(show_plot=True, save_plot=False)
    psnr_metric.update(reconstructed_images, observed_images)
    psnr_value = psnr_metric.compute()

    print(f"PSNR: {psnr_value}")
