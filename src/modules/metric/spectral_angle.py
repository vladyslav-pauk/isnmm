import itertools
import torch
import torchmetrics

from src.utils.matrix_tools import spectral_angle_mapper, match_components


class SpectralAngle(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, degrees=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("sum_spectral_angle", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

        self.degrees = degrees

    def update(self, matrix_est, matrix_true):
        # matrix_true_norm = matrix_true / torch.norm(matrix_true, dim=0, keepdim=True)
        # matrix_est_norm = matrix_est / torch.norm(matrix_est, dim=0, keepdim=True)
        #
        # cosines = torch.sum(matrix_true_norm * matrix_est_norm, dim=0)
        # spectral_angle = torch.acos(cosines).mean()

        spectral_angle = spectral_angle_mapper(match_components(matrix_true, matrix_est), matrix_true)
        self.sum_spectral_angle += spectral_angle * matrix_est.shape[0]
        self.count += matrix_est.shape[0]

    def compute(self):
        spectral_angle = self.sum_spectral_angle / self.count
        self.sum_spectral_angle = torch.tensor(0.0)
        self.count = torch.tensor(0)

        if self.degrees:
            return spectral_angle * 180 / torch.pi
        return spectral_angle
