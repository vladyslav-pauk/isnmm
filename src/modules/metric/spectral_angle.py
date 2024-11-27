import torch
import torchmetrics

from src.utils.matrix_tools import spectral_angle_mapper, match_components
from src.modules.utils import unmix, permute


class SpectralAngle(torchmetrics.Metric):
    def __init__(self, unmixing=None, dist_sync_on_step=False, degrees=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("matrix_true", default=[], dist_reduce_fx="cat")
        self.add_state("matrix_est", default=[], dist_reduce_fx="cat")

        self.degrees = degrees
        self.tensor = None
        self.unmixing = unmixing

    def update(self, matrix_est=None, matrix_true=None):
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
        # state_data = permute(state_data)

        # matrix_true_norm = matrix_true / torch.norm(matrix_true, dim=0, keepdim=True)
        # matrix_est_norm = matrix_est / torch.norm(matrix_est, dim=0, keepdim=True)
        #
        # cosines = torch.sum(matrix_true_norm * matrix_est_norm, dim=0)
        # spectral_angle = torch.acos(cosines).mean()

        spectral_angle = spectral_angle_mapper(match_components(state_data["true"], state_data["latent_sample"]), state_data["true"])
        if self.degrees:
            spectral_angle = spectral_angle * 180 / torch.pi

        return spectral_angle
