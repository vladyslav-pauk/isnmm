import torch
import torchmetrics

from src.modules.data.hyperspectral import DataModule
from src.modules.transform.convolution import HyperspectralTransform
from src.modules.utils import unmix, permute


class Hyperspectral(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, latent_dim=None, unmixing=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.unmixing = unmixing
        self.latent_dim = latent_dim
        self.state_data = {}
        self.tensor = None
        self.tensors = {}

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.state_data:
                self.state_data[key] = []
            self.state_data[key].append(value.clone().detach().cpu())

    def compute(self):
        state_data = {}
        for key, value in self.state_data.items():
            state_data[key] = torch.cat(value, dim=0)

        state_data = unmix(state_data, self.unmixing, self.latent_dim)
        state_data = permute(state_data)

        self.tensors = state_data
        self.state_data.clear()
        return None


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

    hyperspectral_metric = Hyperspectral()
    hyperspectral_metric.update(recovered_abundances=reconstructed_images, transformed_images=transformed_images)
    hyperspectral_metric.compute()

# todo: rewrite using torch add_state interface
# todo: use common interface for tensor & state_data for all metrics, refactor compute, update, etc
