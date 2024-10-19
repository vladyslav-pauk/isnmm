from torch import zeros
from torch.distributions.transforms import Transform
from torch.distributions import constraints


class Identity(Transform):
    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @property
    def domain(self):
        return constraints.simplex

    @property
    def codomain(self):
        return constraints.real_vector

    def __call__(self, sample):
        return sample

    def _inverse(self, sample):
        return sample

    def log_abs_det_jacobian(self, z, y):
        return zeros(z.shape[0], device=z.device)

