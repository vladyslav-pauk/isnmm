import torch
import torchmetrics

from src.utils.matrix_tools import spectral_angle_mapper, match_components


class SpectralAngle(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, degrees=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("true", default=[], dist_reduce_fx="cat")
        self.add_state("estimated", default=[], dist_reduce_fx="cat")

        self.degrees = degrees
        self.tensors = {}

    def update(self, estimated=None, true=None):
        self.true.append(true)
        self.estimated.append(estimated)

    def compute(self):
        estimated = torch.cat(self.estimated, dim=0)
        true = torch.cat(self.true, dim=0)

        # matrix_true_norm = matrix_true / torch.norm(matrix_true, dim=0, keepdim=True)
        # matrix_est_norm = matrix_est / torch.norm(matrix_est, dim=0, keepdim=True)
        #
        # cosines = torch.sum(matrix_true_norm * matrix_est_norm, dim=0)
        # spectral_angle = torch.acos(cosines).mean()

        spectral_angle = spectral_angle_mapper(match_components(true, estimated), true)
        if self.degrees:
            spectral_angle = spectral_angle * 180 / torch.pi

        return spectral_angle

    def plot(self, image_dims, show_plot=False, save_plot=False):
        pass
