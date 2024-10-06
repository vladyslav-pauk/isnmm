import torch
import torchmetrics


class MatrixChange(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("prev_value", default=torch.tensor(torch.inf), dist_reduce_fx="mean")

        self.relative_change = torch.tensor(torch.inf)
        self.prev_value = torch.tensor(torch.inf)

    def update(self, current_value):
        if not torch.isnan(self.prev_value).any():

            value_change = current_value - self.prev_value

            norm = lambda x: (torch.sum(x ** 2) / x.numel()).pow(0.5)

            self.relative_change = norm(value_change) / norm(current_value)

        self.prev_value = current_value

    def compute(self):
        return self.relative_change # 10 * torch.log10(self.relative_change + 1e-8)


    # def update(self, current_value):
    #     if not torch.isnan(self.prev_value).any():
    #         value_change = current_value - self.prev_value
    #         self.relative_change = self._criterion(value_change) / self._criterion(self.prev_value)
    #     self.prev_value = current_value
    #
    # def _criterion(self, x):
    #     # norm = torch.linalg.matrix_norm(x, ord='fro')
    #     norm = (torch.sum(x ** 2) / x.numel()).pow(0.5)
    #     return norm
