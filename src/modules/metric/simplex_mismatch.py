import torch
import torchmetrics
from src.utils.plot_tools import plot_image


class SimplexMismatch(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, db=False, rmse=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.db = db
        self.rmse = rmse

        self.add_state("error", default=[], dist_reduce_fx='cat')

        self.tensors = {}

    def update(self, estimated=None, true=None):
        mse = 1 - torch.sum(estimated, dim=-1, keepdim=True)
        self.error.append(mse)

    def compute(self):
        error = torch.cat(self.error, dim=0)
        if self.rmse:
            error = error.sqrt()
        self.tensors = {"error": error}

        mean_error = error.mean()
        if self.db:
            mean_error = 10 * torch.log10(mean_error)

        return mean_error

    def plot(self, image_dims, show_plot=False, save_plot=False):
        plt, axes = plot_image(
            tensors=self.tensors,
            image_dims=image_dims,
            show_plot=show_plot,
            save_plot=save_plot
        )
        return plt, axes

# fixme: instead of Tensor metric plot_components in mse. tensors for !=2
