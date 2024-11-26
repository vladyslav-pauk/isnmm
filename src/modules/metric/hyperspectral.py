import torch
import torchmetrics

from src.modules.data.hyperspectral import DataModule
from src.modules.transform.convolution import HyperspectralTransform
from src.modules.utils import plot_data, unmix


# todo: rewrite using torch add_state interface


class Hyperspectral(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, show_plot=False, log_plot=True, save_plot=True, image_dims=None,
                 unmixing=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.show_plot = show_plot
        self.log_plot = log_plot
        self.save_plot = save_plot
        self.image_dims = image_dims

        self.unmixing = unmixing
        self.state_data = {}
        self.tensor = None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.state_data:
                self.state_data[key] = []
            self.state_data[key].append(value.clone().detach().cpu())

    def compute(self):
        state_data = {}
        for key, value in self.state_data.items():
            state_data[key] = torch.cat(value, dim=0)

        state_data = self.unmix(state_data)
        state_data = self.permute(state_data)

        data = {key: val for key, val in state_data.items() if key != 'labels'}
        plot_data(data, self.image_dims, show_plot=self.show_plot, save_plot=self.save_plot)
        self.state_data.clear()
        return None

    def unmix(self, state_data):
        if self.unmixing and "latent_sample" in state_data:
            state_data["latent_sample"], mixing_matrix = unmix(
                state_data["latent_sample"],
                latent_dim=self.image_dims[0],
                model=self.unmixing
            )
            mixing_matrix_pinv = torch.linalg.pinv(mixing_matrix)

            for key, value in state_data.items():
                if key != "latent_sample" and key != "true":
                    state_data[key] = torch.matmul(mixing_matrix_pinv, value.T).T

        return state_data

    def permute(self, state_data):
        if "latent_sample" in state_data:
            permutation, _ = self.best_permutation_mse(state_data["latent_sample"], state_data["true"])
            for key in state_data:
                if key != "true":
                    state_data[key] = state_data[key][:, permutation]
        return state_data

    def best_permutation_mse(self, model_A, true_A):
        import itertools
        col_permutations = itertools.permutations(range(model_A.size(1)))
        best_mse = float('inf')

        for perm in col_permutations:


            permuted_model_A = model_A[:, list(perm)]
            mean_mse = torch.mean((true_A - permuted_model_A).pow(2))
            mse = (true_A - permuted_model_A).pow(2)

            if mean_mse < best_mse:
                permutation = list(perm)
                best_mse = mean_mse

        return permutation, best_mse
    # fixme: factor out unmixing and permutation to experiments?


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
