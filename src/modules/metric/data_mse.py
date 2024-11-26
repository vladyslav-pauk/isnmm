import torch
import torchmetrics
import itertools


import torch
import torchmetrics
import itertools

from src.modules.utils import plot_data


class DataMse(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, db=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.db = db
        self.add_state("model_data", default=[], dist_reduce_fx='cat')
        self.add_state("true_data", default=[], dist_reduce_fx='cat')

        self.tensor = None

    def update(self, matrix_true=None, matrix_est=None):

        self.model_data.append(matrix_est.clone().detach().cpu())
        self.true_data.append(matrix_true.clone().detach().cpu())

    def compute(self):
        model_data = torch.cat(self.model_data, dim=0)
        true_data = torch.cat(self.true_data, dim=0)

        mean_mse, mse = self.best_permutation_mse(model_data, true_data)

        if self.db:
            mean_mse = 10 * torch.log10(mean_mse)

        # data = {key: val for key, val in mse if key != 'labels'}
        # data = {key: torch.cat(val, dim=0) for key, val in self.state_data.items() if key != 'labels'}
        # plot_data(mse, self.image_dims, show_plot=self.show_plot, save_plot=self.save_plot)
        self.tensor = mse
        return mean_mse
        # fixme: final metrics wrong, not from the last checkpoint in run hyperspectral

    # def best_permutation_mse(self, model_A, true_A):
    #     col_permutations = itertools.permutations(range(model_A.size(1)))
    #     best_mse = float('inf')
    #     # best_mean_mse = torch.tensor(float('inf'))
    #
    #     for perm in col_permutations:
    #         permuted_model_A = model_A[:, list(perm)]
    #         mse = (true_A - permuted_model_A).pow(2)
    #         mean_mse = torch.mean(mse)
    #
    #         if mean_mse < best_mse:
    #             best_mean_mse = mean_mse
    #             best_mse = mse
    #
    #     return best_mean_mse, best_mse

    def best_permutation_mse(self, model_A, true_A):
        col_permutations = itertools.permutations(range(model_A.size(1)))
        best_mse = float('inf')

        for perm in col_permutations:
            permuted_model_A = model_A[:, list(perm)]
            mean_mse = torch.mean((true_A - permuted_model_A).pow(2))
            mse = (true_A - permuted_model_A).pow(2)

            if mean_mse < best_mse:
                best_mse = mean_mse

        return best_mse, mse.detach().cpu()
