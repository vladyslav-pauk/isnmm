import torch
import torchmetrics

from src.modules.utils import unmix, permute


class DataMse(torchmetrics.Metric):
    def __init__(self, unmixing=None, dist_sync_on_step=False, db=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.db = db
        self.add_state("matrix_true", default=[], dist_reduce_fx='cat')
        self.add_state("matrix_est", default=[], dist_reduce_fx='cat')

        self.tensor = None
        self.unmixing = unmixing

    def update(self, matrix_true=None, matrix_est=None):
        self.matrix_true.append(matrix_true)
        self.matrix_est.append(matrix_est)

    def compute(self):
        state_data = {
            "latent_sample": self.matrix_est,
            "true": self.matrix_true
        }
        for key, value in state_data.items():
            state_data[key] = torch.cat(value, dim=0)

        state_data = unmix(state_data, self.unmixing, state_data["true"].shape[-1])
        state_data = permute(state_data)

        mse = (state_data["latent_sample"] - state_data["true"]).pow(2)
        mean_mse = mse.mean()

        if self.db:
            mean_mse = 10 * torch.log10(mean_mse)

        # data = {key: val for key, val in mse if key != 'labels'}
        # data = {key: torch.cat(val, dim=0) for key, val in self.state_data.items() if key != 'labels'}
        # plot_data(mse, self.image_dims, show_plot=self.show_plot, save_plot=self.save_plot)
        self.tensor = mse
        return mean_mse
