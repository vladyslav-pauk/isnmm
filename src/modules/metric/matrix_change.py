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
            self.relative_change = self._criterion(value_change) / self._criterion(current_value)

        self.prev_value = current_value.clone()

    def _criterion(self, value):
        # criterion = torch.linalg.matrix_norm(value, ord='fro')
        criterion = (torch.sum(value ** 2) / value.numel()).pow(0.5)
        return criterion

    def compute(self):
        return self.relative_change # 10 * torch.log10(self.relative_change + 1e-8)
