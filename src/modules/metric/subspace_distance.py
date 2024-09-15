import torch
import torchmetrics


# fixme: test independently subspace distance
# todo: find similar probabilistic measure and use IS expectation.
class SubspaceDistance(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("max_singular_value", default=torch.tensor(0.0), dist_reduce_fx="max")

    def update(self, true_U, model_U):
        true_U_pseudo_inv = torch.linalg.pinv(true_U)

        I = torch.eye(true_U.shape[-1], device=true_U.device)
        P_true_orth = I - true_U_pseudo_inv @ true_U

        matrix_product = model_U @ P_true_orth
        singular_values = torch.linalg.svd(matrix_product)[1]

        self.max_singular_value = torch.max(self.max_singular_value, singular_values.max())

    def compute(self):
        return self.max_singular_value
