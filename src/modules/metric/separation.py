import torch
import torchmetrics
from src.utils.plot_tools import plot_image


class Separation(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, db=False, rmse=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.db = db
        self.rmse = rmse

        self.add_state("separation", default=[], dist_reduce_fx='cat')

        self.tensors = {}

    def update(self, estimated=None):
        sparsity_index = (torch.tensor(estimated.shape[-1]).sqrt() - torch.norm(estimated, p=1, dim=-1, keepdim=True) / torch.norm(estimated, p=2, dim=-1, keepdim=True)) / (torch.tensor(estimated.shape[-1]).sqrt() - 1)
        self.separation.append(sparsity_index)

    def compute(self):
        separation = torch.cat(self.separation, dim=0)
        if self.rmse:
            separation = separation.sqrt()
        self.tensors = {"separation": separation}

        mean_separation = separation.mean()
        if self.db:
            mean_separation = 10 * torch.log10(mean_separation)

        return mean_separation

    def plot(self, image_dims, show_plot=False, save_plot=False):
        plt, axes = plot_image(
            tensors=self.tensors,
            image_dims=image_dims,
            show_plot=show_plot,
            save_plot=save_plot
        )
        return plt, axes

# fixme: filter noise, separation only if value above threshold
