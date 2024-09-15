import torch
import torchmetrics


class SpectralAngle(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("sum_spectral_angle", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, model_A, true_A):
        true_A_norm = true_A / torch.norm(true_A, dim=1, keepdim=True)
        model_A_norm = model_A / torch.norm(model_A, dim=1, keepdim=True)
        cosines = torch.sum(true_A_norm * model_A_norm, dim=1)
        spectral_angle = torch.acos(cosines).mean()
        self.sum_spectral_angle += spectral_angle
        self.count += 1

    def compute(self):
        return self.sum_spectral_angle / self.count
