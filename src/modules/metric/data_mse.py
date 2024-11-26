import torch
import torchmetrics
import itertools


import torch
import torchmetrics
import itertools


class DataMse(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, db=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.db = db
        self.add_state("model_data", default=[], dist_reduce_fx='cat')
        self.add_state("true_data", default=[], dist_reduce_fx='cat')

    def update(self, matrix_true=None, matrix_est=None):

        self.model_data.append(matrix_est.clone().detach().cpu())
        self.true_data.append(matrix_true.clone().detach().cpu())

    def compute(self):
        model_data = torch.cat(self.model_data, dim=0)
        true_data = torch.cat(self.true_data, dim=0)

        best_mse = self.best_permutation_mse(model_data, true_data)

        if self.db:
            best_mse = 10 * torch.log10(best_mse)

        return best_mse

    def best_permutation_mse(self, model_A, true_A):
        col_permutations = itertools.permutations(range(model_A.size(1)))
        best_mse = float('inf')

        for perm in col_permutations:
            permuted_model_A = model_A[:, list(perm)]
            mse = torch.mean((true_A - permuted_model_A).pow(2))

            if mse < best_mse:
                best_mse = mse

        return best_mse
