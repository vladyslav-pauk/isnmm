import torch
from torch.optim import Optimizer
from torch.optim import Adam


class ConstrainedLagrangeOptimizer(Optimizer):
    def __init__(self, params=None, lr=None, rho=None, inner_iters=None, n_sample=None, observed_dim=None, constraint_fn=None):
        defaults = dict(lr=lr, rho=rho, inner_iters=inner_iters)
        super(ConstrainedLagrangeOptimizer, self).__init__(params, defaults)

        self.constraint_fn = constraint_fn
        self.latent_sample_buffer = torch.zeros((n_sample, observed_dim))
        self.count_buffer = torch.zeros(n_sample, dtype=torch.int32)
        self.mult = torch.zeros(n_sample)
        self.rho = rho
        self.inner_iters = inner_iters
        self.global_step = 0

        self.base_optimizer = Adam(params, lr=lr)

    def to(self, device):
        self.latent_sample_buffer = self.latent_sample_buffer.to(device)
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
            latent_sample = self.latent_sample_buffer[idxes]

            diff = torch.sum(latent_sample, dim=1) - 1.0
            self.mult[idxes] += self.rho * diff

            self.latent_sample_buffer[idxes] = 0.0
            self.count_buffer[idxes] = 0

    def update_buffers(self, idxes, latent_sample):
        idxes = idxes.to(self.latent_sample_buffer.device)
        latent_sample = latent_sample.to(self.latent_sample_buffer.device).detach()

        self.latent_sample_buffer[idxes] = latent_sample
        self.count_buffer[idxes] += 1

    def compute_constraint_errors(self, fx, idxes, batch):
        batch_size = batch.size(0)
        tmp = self.constraint_fn(fx).to(self.mult.device)
        idxes = idxes.to(self.mult.device)

        mult = self.mult[idxes]
        feasible_err = torch.dot(mult, tmp) / batch_size
        augmented_err = (self.rho / 2) * torch.norm(tmp) ** 2 / batch_size

        return {"feasible": feasible_err.to(fx.device), "augmented": augmented_err.to(fx.device)}
