import torch
import torchmetrics
from scipy.linalg import subspace_angles
import itertools


class SubspaceDistance(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("latent_sample_qr", default=[], dist_reduce_fx='cat')
        self.add_state("latent_sample", default=[], dist_reduce_fx='cat')
        self.tensor = None

    def update(self, sample=None, sample_qr=None):

        latent_sample_qr = sample_qr.detach().cpu()
        latent_sample = sample.detach().cpu()

        self.latent_sample_qr.append(latent_sample_qr)
        self.latent_sample.append(latent_sample)

    def compute(self):
        latent_sample_qr = torch.cat(self.latent_sample_qr)
        latent_sample = torch.cat(self.latent_sample)
        qf, _ = torch.linalg.qr(latent_sample)

        angles = torch.tensor(subspace_angles(latent_sample_qr, qf))
        subspace_dist = torch.sin(angles)

        return torch.sum(subspace_dist.pow(2))

    def match_components(self, matrix_model, matrix_true):
        num_cols = matrix_model.size(1)
        col_permutations = itertools.permutations(range(num_cols))

        best_mse = float('inf')
        best_perm = None

        for perm in col_permutations:
            permuted_matrix = matrix_model[:, list(perm)]
            mse = torch.mean((matrix_true - permuted_matrix).pow(2))

            if mse < best_mse:
                best_mse = mse
                best_perm = permuted_matrix

        return best_perm
