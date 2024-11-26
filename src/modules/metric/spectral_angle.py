import torch
import torchmetrics

from src.utils.matrix_tools import spectral_angle_mapper, match_components


class SpectralAngle(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, degrees=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("matrix_true", default=[], dist_reduce_fx="cat")
        self.add_state("matrix_est", default=[], dist_reduce_fx="cat")

        self.degrees = degrees
        self.tensor = None

    def update(self, matrix_est=None, matrix_true=None):
        self.matrix_true.append(matrix_true)
        self.matrix_est.append(matrix_est)

    def compute(self):
        matrix_true = torch.cat(self.matrix_true)
        matrix_est = torch.cat(self.matrix_est)

        # matrix_true_norm = matrix_true / torch.norm(matrix_true, dim=0, keepdim=True)
        # matrix_est_norm = matrix_est / torch.norm(matrix_est, dim=0, keepdim=True)
        #
        # cosines = torch.sum(matrix_true_norm * matrix_est_norm, dim=0)
        # spectral_angle = torch.acos(cosines).mean()

        spectral_angle = spectral_angle_mapper(match_components(matrix_true, matrix_est), matrix_true)
        if self.degrees:
            spectral_angle = spectral_angle * 180 / torch.pi

        return spectral_angle
