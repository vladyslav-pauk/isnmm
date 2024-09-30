import torch
import torchmetrics


class ConstraintError(torchmetrics.Metric):
    def __init__(self, constraint_fn, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.constraint_fn = constraint_fn
        self.add_state("constraint_violations", default=[], dist_reduce_fx="cat")

    def update(self, F):
        constraint_violation = self.constraint_fn(F)
        self.constraint_violations.append(constraint_violation)

    def compute(self):
        if len(self.constraint_violations) > 0:
            concatenated_constraints = torch.cat(self.constraint_violations, dim=0)

            self.reset()
            return torch.norm(concatenated_constraints) ** 2
        else:
            return torch.tensor(0.0)

    def reset(self):
        self.constraint_violations = []