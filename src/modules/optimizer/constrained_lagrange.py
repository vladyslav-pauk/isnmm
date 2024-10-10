import torch
from torch.optim import Optimizer
from torch.optim import Adam


class ConstrainedLagrangeOptimizer(Optimizer):
    def __init__(self, params=None, constraint_fn=None, optimizer_config=None):
        defaults = dict(**optimizer_config)
        super(ConstrainedLagrangeOptimizer, self).__init__(params, defaults)

        self.rho = optimizer_config['rho']
        self.update_frequency = optimizer_config['update_frequency']

        self.constraint_fn = constraint_fn
        self.global_step = 0
        self.base_optimizer = Adam(params, optimizer_config['lr']["encoder"])

        self._buffers = {}

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self.base_optimizer.step()

        if (self.global_step + 1) % self.update_frequency == 0:
            self.update_multipliers()

        self.global_step += 1
        return loss

    def update_multipliers(self):
        idxes = self._buffers["count"].nonzero(as_tuple=True)[0]
        latent_sample = self._buffers["latent_sample"][idxes]

        diff = torch.sum(latent_sample, dim=1) - 1.0
        self._buffers["mult"][idxes] += self.rho * diff

        self._buffers["latent_sample"][idxes] = 0.0
        self._buffers["count"][idxes] = 0

    def initialize_buffers(self, n_sample, latent_dim):
        self.register_buffer("latent_sample", torch.zeros((n_sample, latent_dim)))
        self.register_buffer("count", torch.zeros(n_sample))
        self.register_buffer("mult", torch.zeros(n_sample))

    def update_buffers(self, idxes, latent_sample):
        idxes = idxes.to(self._buffers["latent_sample"].device)
        latent_sample = latent_sample.to(self._buffers["latent_sample"].device).detach()

        self._buffers["latent_sample"][idxes] = latent_sample
        self._buffers["count"][idxes] += 1

    def compute_regularization_loss(self, latent_sample, observed_sample, idxes):
        if 'mult' not in self._buffers:
            self.initialize_buffers(
                n_sample=observed_sample.shape[0] * self.update_frequency,
                latent_dim=latent_sample.shape[-1]
            )
        batch_size = observed_sample.size(0)
        tmp = self.constraint_fn(latent_sample).to(self._buffers["mult"].device)
        idxes = idxes.to(self._buffers["mult"].device)

        mult = self._buffers["mult"][idxes]

        feasible_err = torch.dot(mult, tmp) / batch_size
        augmented_err = (self.rho / 2) * torch.norm(tmp) ** 2 / batch_size

        return {"feasible": feasible_err.to(latent_sample.device), "augmented": augmented_err.to(latent_sample.device)}

    def register_buffer(self, name, tensor):
        self._buffers[name] = tensor

    def to(self, device):
        for name, buffer in self._buffers.items():
            self._buffers[name] = buffer.to(device)

# todo: refactor so the arguments passed as
#  {'params': self.encoder.parameters(), 'lr': optimizer_config['lr']["encoder"]}
