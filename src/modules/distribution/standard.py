from torch import diag_embed
import torch.distributions as td


class MultivariateNormal(td.MultivariateNormal):
    def __init__(self, loc, scale):
        scale.clamp(min=1e-12)
        super().__init__(loc, diag_embed(scale**2))


class Normal(td.Normal):
    def __init__(self, loc, scale):
        super().__init__(loc, scale)


class Dirichlet(td.Dirichlet):
    def __init__(self, loc, alpha):
        super().__init__(alpha)
