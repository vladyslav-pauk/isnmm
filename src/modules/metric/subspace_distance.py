import torch
import torchmetrics
import scipy.linalg
import itertools

class SubspaceDistance(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # State to store subspace distance over multiple updates
        self.add_state("subspace_distances", default=[], dist_reduce_fx=None)

    def update(self, idxes, latent_sample, latent_sample_qr):
        # Detach and move to CPU
        latent_sample = latent_sample[idxes].detach().cpu()
        latent_sample_qr = latent_sample_qr.detach().cpu()

        # QR decomposition to get orthonormal basis of latent_sample
        qf, _ = torch.linalg.qr(latent_sample)

        # Find best permutation to match components and compute subspace angles
        matched_qf = self.match_components(qf, latent_sample_qr)
        subspace_angles = torch.tensor(scipy.linalg.subspace_angles(latent_sample_qr, matched_qf))
        subspace_dist = torch.sin(subspace_angles)

        # Store the subspace distance
        self.subspace_distances.append(subspace_dist)

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

    def compute(self):
        # Combine all subspace distances and compute their mean
        if len(self.subspace_distances) > 0:
            subspace_dist_tensor = torch.cat(self.subspace_distances)
            return torch.mean(subspace_dist_tensor)
        else:
            return torch.tensor(0.0)  # Return 0 if no data has been accumulated