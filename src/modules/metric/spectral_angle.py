import torch
import torchmetrics


class SpectralAngle(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("sum_spectral_angle", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, matrix_est, matrix_true):
        matrix_true = matrix_true.T
        matrix_est = matrix_est.T
        matrix_true_norm = matrix_true / torch.norm(matrix_true, dim=1, keepdim=True)
        matrix_est_norm = matrix_est / torch.norm(matrix_est, dim=1, keepdim=True)

        cosines = torch.sum(matrix_true_norm * matrix_est_norm, dim=1)
        spectral_angle = torch.acos(cosines).sum()
        self.sum_spectral_angle += spectral_angle
        self.count += cosines.shape[0]

    def compute(self):
        return self.sum_spectral_angle / self.count

    # todo: use function from helpers
