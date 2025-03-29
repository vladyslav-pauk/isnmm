import torch
import torchmetrics
from scipy.linalg import subspace_angles


class SubspaceDistance(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("true_qr", default=[], dist_reduce_fx='cat')
        self.add_state("estimated", default=[], dist_reduce_fx='cat')

        self.tensors = {}

    def update(self, estimated=None, true_qr=None):
        self.true_qr.append(true_qr)
        self.estimated.append(estimated)

    def compute(self):

        estimated = torch.cat(self.estimated, dim=0)
        true_qr = torch.cat(self.true_qr, dim=0)

        estimated_qf, _ = torch.linalg.qr(estimated)

        true_qr = true_qr.detach().cpu().numpy()
        estimated_qf = estimated_qf.detach().cpu().numpy()

        angles = subspace_angles(true_qr, estimated_qf)
        angles = torch.tensor(angles, device=estimated.device)

        subspace_dist = torch.sin(angles)

        return torch.sum(subspace_dist.pow(2))

    def plot(self, image_dims=None, show_plot=False, save_plot=False):
        pass
