import torch
import torchmetrics


class MatrixVolume(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("vol", default=torch.tensor(float('nan')), dist_reduce_fx="mean")

    def update(self, matrix):
        vol = torch.det(matrix.T @ matrix)
        self.vol = matrix

    def compute(self):
        return self.vol
