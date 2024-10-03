import torch
import torchmetrics
import scipy.linalg


class SubspaceDistance(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("max_singular_value", default=torch.tensor(0.0), dist_reduce_fx="max")
        self.subspace_dist = None

    def update(self, idxes, reconstructed_sample, latent_sample):
        reconstructed_sample = reconstructed_sample[idxes].detach().cpu()
        latent_sample = latent_sample.detach().cpu()
        qf, _ = torch.linalg.qr(reconstructed_sample)
        subspace_angles = torch.tensor(scipy.linalg.subspace_angles(latent_sample, qf)[0])
        self.subspace_dist = torch.sin(subspace_angles)

    def compute(self):
        return self.subspace_dist
