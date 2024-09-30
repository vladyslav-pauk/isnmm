import torch
from torch.optim import Optimizer
from torch.optim import Adam


class ConstrainedLagrangeOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, rho=1e2, inner_iters=1, n_sample=100, input_dim=10, constraint_fn=None):
        defaults = dict(lr=lr, rho=rho, inner_iters=inner_iters)
        super(ConstrainedLagrangeOptimizer, self).__init__(params, defaults)

        self.constraint_fn = constraint_fn

        self.F_buffer = torch.zeros((n_sample, input_dim))
        self.count_buffer = torch.zeros(n_sample, dtype=torch.int32)
        self.mult = torch.zeros(n_sample)
        self.rho = rho
        self.inner_iters = inner_iters
        self.global_step = 0

        self.base_optimizer = Adam(params, lr=lr)

    def to(self, device):
        self.F_buffer = self.F_buffer.to(device)
        self.count_buffer = self.count_buffer.to(device)
        self.mult = self.mult.to(device)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self.base_optimizer.step()
        self.update_multipliers()

        self.global_step += 1
        return loss

    def update_multipliers(self):
        if (self.global_step + 1) % self.inner_iters == 0:
            idxes = self.count_buffer.nonzero(as_tuple=True)[0]
            F = self.F_buffer[idxes]
            print(F)
            diff = torch.sum(F, dim=1) - 1.0

            self.mult[idxes] += self.rho * diff

            self.F_buffer[idxes] = 0.0
            self.count_buffer[idxes] = 0

    def compute_constraint_errors(self, fx, idxes, batch):
        batch_size = batch.size(0)

        tmp = self.constraint_fn(fx).to(self.mult.device)
        idxes = idxes.to(self.mult.device)

        mult = self.mult[idxes]
        feasible_err = torch.dot(mult, tmp) / batch_size
        augmented_err = (self.rho / 2) * torch.norm(tmp) ** 2 / batch_size

        return {"feasible": feasible_err.to(fx.device), "augmented": augmented_err.to(fx.device)}