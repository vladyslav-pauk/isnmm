import torch
import torchmetrics
import scipy.linalg


class SubspaceDistance(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("max_singular_value", default=torch.tensor(0.0), dist_reduce_fx="max")

    # def update(self, true_U, model_U):
    #     # Ensure that the inputs are on the same device
    #     true_U, model_U = true_U.to('cpu').detach().numpy(), model_U.to('cpu').detach().numpy()
    #
    #     # QR decomposition to find orthonormal bases of the subspaces
    #     q_true, _ = torch.linalg.qr(torch.tensor(true_U))
    #     q_model, _ = torch.linalg.qr(torch.tensor(model_U))
    #
    #     # Compute subspace angles using scipy's function
    #     subspace_angles = scipy.linalg.subspace_angles(q_true.numpy(), q_model.numpy())
    #
    #     # Take the sine of the largest principal angle as the subspace distance
    #     subspace_dist = torch.sin(torch.tensor(subspace_angles[0]))  # Largest principal angle
    #
    #     # Update the state
    #     self.max_singular_value = torch.max(self.max_singular_value, subspace_dist)

    def update(self, idxes, F, qs):
        F_cpu = F.to('cpu').detach()
        qs = qs.to('cpu').detach()
        qf, _ = torch.linalg.qr(F_cpu)

        import scipy
        self.subspace_dist = torch.sin(torch.tensor(scipy.linalg.subspace_angles(qs, qf)[0]))

    def compute(self):
        return self.subspace_dist