import torch
from torch.distributions import TransformedDistribution


class LocationScale(TransformedDistribution):
    def __init__(self, base_distribution, transform):
        super(LocationScale, self).__init__(
            base_distribution, [transform], validate_args=None
        )

    def log_prob(self, sample):
        transformed_sample = self.transforms[0]._inverse(sample)
        log_prob_base = self.base_dist.log_prob(transformed_sample)
        log_det_jacobian = self.transforms[0].log_abs_det_jacobian(sample, transformed_sample)
        return log_prob_base + log_det_jacobian
