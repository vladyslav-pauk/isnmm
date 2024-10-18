import torch
from torch.distributions import MultivariateNormal, TransformedDistribution, constraints
from src.modules.transform.logit_transform import LogitTransform
from torch.distributions import Distribution, constraints


class TransformedNormal(TransformedDistribution):
    def __init__(self, mean, std, transform, validate_args=None):
        covariance_matrix = torch.diag_embed(std)
        self._mean = mean
        self._std = std
        base_distribution = MultivariateNormal(mean, covariance_matrix)
        self.epsilon = 1e-12
        super(TransformedNormal, self).__init__(base_distribution, [transform], validate_args=validate_args)

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def log_prob(self, z):
        transformed_z = self.transforms[0](z)
        log_prob_base = self.base_dist.log_prob(transformed_z)
        log_det_jacobian = self.transforms[0].log_abs_det_jacobian(z, transformed_z)
        return log_prob_base + log_det_jacobian


class LogisticNormal(Distribution):
    arg_constraints = {'mean': constraints.real, 'std': constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, mean, std, validate_args=None, **kwargs):
        self.mean = mean
        self.std = std
        self.epsilon = 1e-12
        super().__init__(validate_args=validate_args)

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value):
        self._mean = value

    @property
    def std(self):
        return self._std

    @std.setter
    def std(self, value):
        self._std = value

    def log_prob(self, latent_sample):
        log_2pi = torch.log(torch.tensor(2 * torch.pi))
        log_var = torch.log(self.std + self.epsilon).unsqueeze(0)
        inv_std = 1 / (self.std + self.epsilon).unsqueeze(0)

        projected_latent = torch.log(latent_sample[..., :-1]) - torch.log(latent_sample[..., -1:])
        centered_latent = projected_latent - self.mean.unsqueeze(0)

        mahalanobis_dist = (centered_latent ** 2 * inv_std).sum(dim=-1)
        log_det_cov = log_var.sum(dim=-1)
        log_prob_base = - 0.5 * (mahalanobis_dist + log_det_cov + (latent_sample.size(-1) - 1) * log_2pi)
        # mahalanobis_dist = torch.einsum('bi,ij,bj->b', diff, torch.inverse(self.covariance_matrix), diff)
        # log_det_cov = torch.logdet(self.covariance_matrix)
        log_jacobian = torch.log(latent_sample).sum(dim=-1)
        # print(log_prob_base)
        # import sys
        # sys.exit()

        return log_prob_base - log_jacobian
