import torch
import torchmetrics
import itertools

import torch
import torchmetrics
import itertools

from src.modules.utils import plot_data


class DataMse(torchmetrics.Metric):
    def __init__(self, unmixing=None, dist_sync_on_step=False, db=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.db = db
        self.add_state("model_data", default=[], dist_reduce_fx='cat')
        self.add_state("true_data", default=[], dist_reduce_fx='cat')

        self.tensor = None
        self.unmixing = unmixing

    def update(self, matrix_true=None, matrix_est=None):
        self.model_data.append(matrix_est.clone().detach().cpu())
        self.true_data.append(matrix_true.clone().detach().cpu())

    def compute(self):
        state_data = {
            "latent_sample": self.model_data,
            "true_data": self.true_data
        }
        for key, value in state_data.items():
            state_data[key] = torch.cat(value, dim=0)

        state_data, mean_mse, mse = self.unmix(state_data)

        if self.db:
            mean_mse = 10 * torch.log10(mean_mse)

        # data = {key: val for key, val in mse if key != 'labels'}
        # data = {key: torch.cat(val, dim=0) for key, val in self.state_data.items() if key != 'labels'}
        # plot_data(mse, self.image_dims, show_plot=self.show_plot, save_plot=self.save_plot)
        self.tensor = mse
        return mean_mse

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
        best_mean_mse = float('inf')
        best_mse = torch.tensor(float('inf'))

        for perm in col_permutations:
            permuted_model_A = model_A[:, list(perm)]
            mean_mse = torch.mean((true_A - permuted_model_A).pow(2))
            mse = (true_A - permuted_model_A).pow(2)

            if mean_mse < best_mean_mse:
                permutation = list(perm)
                best_mean_mse = mean_mse
                best_mse = mse

        return permutation, best_mean_mse, best_mse.detach().cpu()
    # fixme: factor out permutation

    # def unmix(self, state_data):
    #     from src.modules.utils import unmix
    #     for key, value in state_data.items():
    #         if key == "model_data":
    #             self.model_data, mixing_matrix = unmix(
    #                 self.model_data,
    #                 latent_dim=self.image_dims[0],
    #                 model=self.unmixing)
    #             mixing_matrix_pinv = torch.linalg.pinv(mixing_matrix)
    #             state_data[key] = torch.matmul(mixing_matrix_pinv, value.T).T
    #     return state_data

    def unmix(self, state_data):
        from src.modules.utils import unmix
        for key, value in state_data.items():
            if key == "latent_sample":
                if self.unmixing:
                    state_data["latent_sample"], mixing_matrix = unmix(
                        state_data["latent_sample"],
                        latent_dim=state_data["true_data"].shape[-1],
                        model=self.unmixing)
                    mixing_matrix_pinv = torch.linalg.pinv(mixing_matrix)

                permutation, mean_mse, mse = self.best_permutation_mse(
                    state_data["latent_sample"],
                    state_data["true_data"]
                )

                state_data["latent_sample"] = state_data["latent_sample"][:, permutation]
                if self.unmixing:
                    mixing_matrix_pinv = mixing_matrix_pinv[permutation]
            elif key != "true_data":
                state_data[key] = torch.matmul(mixing_matrix_pinv, value.T).T

        return state_data, mean_mse, mse
