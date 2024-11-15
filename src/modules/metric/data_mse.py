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
        # States to collect model and true data across batches
        self.add_state("model_data", default=[], dist_reduce_fx=None)
        self.add_state("true_data", default=[], dist_reduce_fx=None)

    def update(self, model_A, true_A):
        # Collect model_A and true_A for all batches
        self.model_data.append(model_A.clone().detach().cpu())
        self.true_data.append(true_A.clone().detach().cpu())

    def compute(self):
        model_data = torch.cat(self.model_data, dim=0)
        true_data = torch.cat(self.true_data, dim=0)

        best_mse = self.find_best_permutation_mse(model_data, true_data)

        if self.db:
            return 10 * torch.log10(best_mse)
        else:
            return best_mse

    def find_best_permutation_mse(self, model_A, true_A):
        num_cols = model_A.size(1)
        col_permutations = itertools.permutations(range(num_cols))

        best_mse = float('inf')

        for perm in col_permutations:
            permuted_model_A = model_A[:, list(perm)]
            mse = torch.mean((true_A - permuted_model_A).pow(2))

            if mse < best_mse:
                best_mse = mse

        return best_mse
