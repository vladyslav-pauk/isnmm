import torch
from torch.distributions import MultivariateNormal, TransformedDistribution, constraints
# from src.modules.transform.logit_transform import LogitTransform
# from torch.distributions import Distribution, constraints


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


# class LogisticNormal(Distribution):
#     arg_constraints = {'mean': constraints.real, 'std': constraints.positive}
#     support = constraints.real
#     has_rsample = True
#
#     def __init__(self, mean, std, validate_args=None, **kwargs):
#         self.mean = mean
#         self.std = std
#         self.epsilon = 1e-12
#         super().__init__(validate_args=validate_args)
#
#     @property
#     def mean(self):
#         return self._mean
#
#     @mean.setter
#     def mean(self, value):
#         self._mean = value
#
#     @property
#     def std(self):
#         return self._std
#
#     @std.setter
#     def std(self, value):
#         self._std = value
#
#     def log_prob(self, latent_sample):
#         log_2pi = torch.log(torch.tensor(2 * torch.pi))
#         log_var = torch.log(self.std + self.epsilon).unsqueeze(0)
#         inv_std = 1 / (self.std + self.epsilon).unsqueeze(0)
#
#         projected_latent = torch.log(latent_sample[..., :-1]) - torch.log(latent_sample[..., -1:])
#         centered_latent = projected_latent - self.mean.unsqueeze(0)
#
#         mahalanobis_dist = (centered_latent ** 2 * inv_std).sum(dim=-1)
#         log_det_cov = log_var.sum(dim=-1)
#         log_prob_base = - 0.5 * (mahalanobis_dist + log_det_cov + (latent_sample.size(-1) - 1) * log_2pi)
#         # mahalanobis_dist = torch.einsum('bi,ij,bj->b', diff, torch.inverse(self.covariance_matrix), diff)
#         # log_det_cov = torch.logdet(self.covariance_matrix)
#         # print(base_log_prob[0][0])
#         # import sys
#         # sys.exit()
#
#         log_jacobian = torch.log(latent_sample).sum(dim=-1)
#         print(log_prob_base)
#         import sys
#         sys.exit()
#
#         return log_prob_base - log_jacobian

# import torch
# from torch.distributions import MultivariateNormal, TransformedDistribution
# from torch.distributions.transforms import Transform
# from torch.distributions import constraints
#
#
# class LogisticNormal(TransformedDistribution):
#     def __init__(self, mean, covariance_matrix, validate_args=None):
#         """
#         Logistic Normal distribution modeled as a transformed distribution based on the
#         logit transformation and a multivariate normal base distribution.
#
#         Args:
#             mean (Tensor): Mean vector of the distribution in log-ratio (logit) space (D-1 dimensional).
#             covariance_matrix (Tensor): Covariance matrix in log-ratio space (D-1 x D-1).
#         """
#         # Define the base distribution as a multivariate normal in log-ratio space (D-1 dimensions)
#         base_distribution = MultivariateNormal(mean, covariance_matrix)
#
#         # Use the Logit transformation as the transformation from the simplex to Euclidean space
#         transform = LogitTransform()
#
#         # Call the parent class with the base distribution and the transformation
#         super(LogisticNormal, self).__init__(base_distribution, transform, validate_args=validate_args)
#
#     def entropy(self):
#         """Manually compute the entropy of the Logistic Normal distribution."""
#         base_entropy = self.base_dist.entropy()  # Entropy of the multivariate normal in logit space
#
#         # Jacobian correction: we add the expected value of the Jacobian correction term (-sum(log(z)))
#         correction_term = -torch.digamma(torch.tensor(self.base_dist.mean.size(-1) + 1))  # Digamma approximation
#
#         return base_entropy + correction_term
#
#
# class LogitTransform(Transform):
#     """Logit transformation (for Logistic Normal distribution) with numerical stability."""
#
#     def __init__(self, cache_size=1):
#         super().__init__(cache_size=cache_size)
#
#     @property
#     def sign(self):
#         return 1
#
#     @property
#     def domain(self):
#         return constraints.simplex
#
#     @property
#     def codomain(self):
#         return constraints.real_vector
#
#     def _call(self, z):
#         """Applies the logit transformation to map from the simplex to Euclidean space."""
#         epsilon = 1e-12  # Small value to avoid log(0)
#         z = torch.clamp(z, min=epsilon, max=1 - epsilon)  # Ensure values are within (0, 1)
#         # Log-ratio transformation: take log of first D-1 components relative to last component
#         log_ratio = torch.log(z[..., :-1]) - torch.log(z[..., -1:])
#         return log_ratio
#
#     def _inverse(self, y):
#         """Applies the inverse logit (softmax) transformation to map back from Euclidean space to the simplex."""
#         exp_y = torch.exp(y)
#         exp_y_sum = torch.sum(exp_y, dim=-1, keepdim=True)
#         last_component = 1.0 / (1.0 + exp_y_sum)
#         return torch.cat([exp_y * last_component, last_component], dim=-1)  # Softmax transformation
#
#     def log_abs_det_jacobian(self, z, y):
#         """Computes the Jacobian of the logit transformation (log determinant)."""
#         epsilon = 1e-12
#         z = torch.clamp(z, min=epsilon)  # Avoid log(0) in Jacobian
#         return -torch.sum(torch.log(z), dim=-1)
