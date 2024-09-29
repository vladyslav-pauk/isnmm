import torch
import torchmetrics


class MatrixVolume(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("vol", default=torch.tensor(float('nan')), dist_reduce_fx="mean")

        self.vol = torch.tensor([])

    def update(self, matrix):
        gamma = torch.lgamma(torch.tensor(matrix.size(1))).exp()
        vol = 1 / gamma * torch.det(matrix.T @ matrix).abs().sqrt()
        self.vol = vol

    def compute(self):
        return self.vol.log()
