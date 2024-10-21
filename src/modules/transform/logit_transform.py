import torch
from torch.distributions import constraints
from torch.distributions.transforms import Transform


class LogitTransform(Transform):
    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @property
    def domain(self):
        return constraints.simplex

    @property
    def codomain(self):
        return constraints.real_vector

    # def __call__(self, sample):
    #     exp_y = torch.exp(sample)
    #     last_component = 1.0 / (1.0 + torch.sum(exp_y, dim=-1, keepdim=True))
    #
    #     return torch.cat([exp_y * last_component, last_component], dim=-1)

    def __call__(self, sample):
        sample_padded = torch.cat([sample, torch.zeros_like(sample[..., :1])], dim=-1)
        lse = torch.logsumexp(sample_padded, dim=-1, keepdim=True)
        transformed_sample = torch.exp(sample_padded - lse)
        return transformed_sample / transformed_sample.sum(dim=-1, keepdim=True)

    def _inverse(self, z):
        epsilon = 1e-12
        z = torch.clamp(z, min=epsilon, max=1 - epsilon)
        log_ratio = torch.log(z[..., :-1]) - torch.log(z[..., -1:])
        return log_ratio

    def log_abs_det_jacobian(self, z, y):
        epsilon = 1e-12
        z = torch.clamp(z, min=epsilon)
        return -torch.sum(torch.log(z), dim=-1)

