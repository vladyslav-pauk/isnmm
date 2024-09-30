import torch
import torchmetrics
import scipy.linalg


class SubspaceDistance(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("max_singular_value", default=torch.tensor(0.0), dist_reduce_fx="max")

    def update(self, idxes, reconstructed_sample, latent_sample):
        reconstructed_sample_cpu = reconstructed_sample[idxes.to('cpu').detach()].to('cpu').detach()
        latent_sample = latent_sample.to('cpu').detach()
        qf, _ = torch.linalg.qr(reconstructed_sample_cpu)
        self.subspace_dist = torch.sin(torch.tensor(scipy.linalg.subspace_angles(latent_sample, qf)[0]))

    def compute(self):
        return self.subspace_dist