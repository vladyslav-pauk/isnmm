import torch
import torchmetrics
import itertools


class MatrixMse(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, db=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.db = db
        self.add_state("min_mse", default=torch.tensor(float('inf')), dist_reduce_fx="min")

    def update(self, model_A, true_A):
        best_mse = self.find_best_permutation_mse(model_A, true_A)
        self.min_mse = torch.min(self.min_mse, best_mse)

    def compute(self):
        if self.db:
            return 10 * torch.log10(self.min_mse)
        else:
            return self.min_mse

    def find_best_permutation_mse(self, model_A, true_A):
        num_cols = model_A.size(1)
        col_permutations = itertools.permutations(range(num_cols))

        best_mse = float('inf')

        for perm in col_permutations:
            permuted_model_A = model_A[:, list(perm)]
            mse = torch.mean(torch.sum((true_A - permuted_model_A) ** 2, dim=1))

            if mse < best_mse:
                best_mse = mse

        return best_mse