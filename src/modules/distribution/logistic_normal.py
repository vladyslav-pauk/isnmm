import torch
from torch.distributions import MultivariateNormal, TransformedDistribution
from torch.distributions.transforms import Transform
from torch.distributions import constraints


class LogitTransform(Transform):
    """Logit transformation (for Logistic Normal distribution) with numerical stability."""

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @property
    def sign(self):
        return 1

    @property
    def domain(self):
        return constraints.simplex

    @property
    def codomain(self):
        return constraints.real_vector

    def _call(self, z):
        """Applies the logit transformation to map from the simplex to Euclidean space."""
        epsilon = 1e-12  # Small value to avoid log(0)
        z = torch.clamp(z, min=epsilon, max=1 - epsilon)  # Ensure values are within (0, 1)
        # Log-ratio transformation: take log of first D-1 components relative to last component
        log_ratio = torch.log(z[..., :-1]) - torch.log(z[..., -1:])
        return log_ratio

    def _inverse(self, y):
        """Applies the inverse logit (softmax) transformation to map back from Euclidean space to the simplex."""
        exp_y = torch.exp(y)
        exp_y_sum = torch.sum(exp_y, dim=-1, keepdim=True)
        last_component = 1.0 / (1.0 + exp_y_sum)
        return torch.cat([exp_y * last_component, last_component], dim=-1)  # Softmax transformation

    def log_abs_det_jacobian(self, z, y):
        """Computes the Jacobian of the logit transformation (log determinant)."""
        epsilon = 1e-12
        z = torch.clamp(z, min=epsilon)  # Avoid log(0) in Jacobian
        return -torch.sum(torch.log(z), dim=-1)


class LogisticNormal(TransformedDistribution):
    def __init__(self, mean, covariance_matrix, validate_args=None):
        """
        Logistic Normal distribution modeled as a transformed distribution based on the
        logit transformation and a multivariate normal base distribution.

        Args:
            mean (Tensor): Mean vector of the distribution in log-ratio (logit) space (D-1 dimensional).
            covariance_matrix (Tensor): Covariance matrix in log-ratio space (D-1 x D-1).
        """
        # Define the base distribution as a multivariate normal in log-ratio space (D-1 dimensions)
        base_distribution = MultivariateNormal(mean, covariance_matrix)

        # Use the Logit transformation as the transformation from the simplex to Euclidean space
        transform = LogitTransform()

        # Call the parent class with the base distribution and the transformation
        super(LogisticNormal, self).__init__(base_distribution, transform, validate_args=validate_args)

    def entropy(self):
        """Manually compute the entropy of the Logistic Normal distribution."""
        base_entropy = self.base_dist.entropy()  # Entropy of the multivariate normal in logit space

        # Jacobian correction: we add the expected value of the Jacobian correction term (-sum(log(z)))
        correction_term = -torch.digamma(torch.tensor(self.base_dist.mean.size(-1) + 1))  # Digamma approximation

        return base_entropy + correction_term


# Example usage:
D = 3  # Number of components in the simplex
mean = torch.zeros(D - 1)  # Mean vector in logit space (D-1 dimensional)
covariance_matrix = torch.eye(D - 1)  # Identity covariance in logit space (D-1 x D-1)

# Create the Logistic Normal distribution using TransformedDistribution
logistic_normal_dist = LogisticNormal(mean, covariance_matrix)

# Sample from the Logistic Normal distribution
samples = logistic_normal_dist.sample((5,))
print("Samples from Logistic Normal distribution on the simplex:")
print(samples)

# Compute log probability for a sample
log_prob = logistic_normal_dist.log_prob(samples)
print("\nLog-probability of the samples:")
print(log_prob)

# Entropy of the Logistic Normal distribution
entropy = logistic_normal_dist.entropy()
print("\nEntropy of the Logistic Normal distribution:")
print(entropy)