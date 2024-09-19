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
            self.relative_change = torch.linalg.matrix_norm(current_value - self.prev_value) / torch.linalg.matrix_norm(current_value)
        self.prev_value = current_value

    def compute(self):
        return self.relative_change
