import torch
import torchmetrics
from scipy.linalg import subspace_angles
import itertools

from src.modules.utils import unmix


class SubspaceDistance(torchmetrics.Metric):
    def __init__(self, unmixing=None, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("latent_sample_qr", default=[], dist_reduce_fx='cat')
        self.add_state("latent_sample", default=[], dist_reduce_fx='cat')

        self.tensor = None
        self.unmixing = unmixing

    def update(self, sample=None, sample_qr=None):

        latent_sample_qr = sample_qr.detach().cpu()
        latent_sample = sample.detach().cpu()

        self.latent_sample_qr.append(latent_sample_qr)
        self.latent_sample.append(latent_sample)

    def compute(self):

        state_data = {
            "latent_sample": self.latent_sample,
            "true": self.latent_sample_qr
        }
        for key, value in state_data.items():
            state_data[key] = torch.cat(value, dim=0)

        qf, _ = torch.linalg.qr(state_data["latent_sample"])

        state_data = unmix(state_data, self.unmixing, state_data["true"].shape[-1])

        angles = torch.tensor(subspace_angles(state_data["true"], qf))
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
