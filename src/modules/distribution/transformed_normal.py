import torch
from torch.distributions import MultivariateNormal, TransformedDistribution


class TransformedNormal(TransformedDistribution):
    def __init__(self, mean, std, transform, validate_args=None):
        covariance_matrix = torch.diag_embed(std)
        self._mean = mean
        self._std = std
        base_distribution = MultivariateNormal(mean, covariance_matrix)
        self.epsilon = 1e-12
        super(TransformedNormal, self).__init__(
            base_distribution, [transform], validate_args=validate_args
        )

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def log_prob(self, sample):
        transformed_sample = self.transforms[0](sample)
        log_prob_base = self.base_dist.log_prob(transformed_sample)
        log_det_jacobian = self.transforms[0].log_abs_det_jacobian(sample, transformed_sample)
        return log_prob_base + log_det_jacobian