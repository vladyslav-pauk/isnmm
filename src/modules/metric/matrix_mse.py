import torch
import torchmetrics


class MatrixMse(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("min_mse", default=torch.tensor(float('inf')), dist_reduce_fx="min")

    def update(self, model_A, true_A):
        mse = torch.mean(torch.sum((true_A - model_A) ** 2, dim=1))
        self.min_mse = torch.min(self.min_mse, mse)

    def compute(self):
        return 10 * torch.log10(self.min_mse)
