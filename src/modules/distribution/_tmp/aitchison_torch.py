import torch
from torch.distributions import MultivariateNormal, TransformedDistribution
from torch.distributions.transforms import Transform
from torch.distributions import constraints


class CLRTransform(Transform):
    """Centered log-ratio (CLR) transformation with numerical stability."""

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
        """Applies the CLR transformation to map from the simplex to Euclidean space."""
        epsilon = 1e-12  # Small value to avoid log(0)
        z = torch.clamp(z, min=epsilon)  # Ensure no values are exactly zero
        geometric_mean = torch.exp(torch.mean(torch.log(z), dim=-1, keepdim=True))
        return torch.log(z / geometric_mean)

    def _inverse(self, y):
        """Applies the inverse CLR transformation to map back from Euclidean space to the simplex."""
        exp_y = torch.exp(y)
        exp_y_sum = torch.sum(exp_y, dim=-1, keepdim=True)
        return torch.clamp(exp_y / torch.clamp(exp_y_sum, min=1e-12), min=1e-12)  # Avoid negative or zero values

    def log_abs_det_jacobian(self, z, y):
        """Computes the Jacobian of the CLR transformation (log determinant)."""
        epsilon = 1e-12
        z = torch.clamp(z, min=epsilon)  # Avoid log(0) in Jacobian
        return -torch.sum(torch.log(z), dim=-1)


class Aitchison(TransformedDistribution):
    def __init__(self, mean, covariance_matrix, validate_args=None):
        """
        Aitchison distribution that operates over the simplex, modeled as a transformed
        distribution based on the CLR transformation and a multivariate normal base distribution.

        Args:
            mean (Tensor): Mean vector of the distribution in log-ratio (CLR) space (D-1 dimensional).
            covariance_matrix (Tensor): Covariance matrix in log-ratio space (D-1 x D-1).
        """
        # Define the base distribution as a multivariate normal in CLR space
        base_distribution = MultivariateNormal(mean, covariance_matrix)

        # Use the CLR transformation as the transformation from the simplex to Euclidean space
        transform = CLRTransform()

        # Call the parent class with the base distribution and the transformation
        super(Aitchison, self).__init__(base_distribution, transform, validate_args=validate_args)

    def entropy(self):
        """Manually compute the entropy of the Aitchison distribution."""
        base_entropy = self.base_dist.entropy()  # Entropy of the multivariate normal in CLR space

        # Jacobian correction: we add the expected value of the Jacobian correction term (-sum(log(z)))
        # In the case of Aitchison, this correction comes from the log of the composition.
        # We'll approximate this correction for a simple case.
        correction_term = -torch.digamma(torch.tensor(self.base_dist.mean.size(-1) + 1))  # Digamma approximation

        return base_entropy + correction_term


# Example usage:
D = 3  # Number of components in the simplex
mean = torch.zeros(D - 1)  # Mean vector in CLR space (D-1 dimensional)
covariance_matrix = torch.eye(D - 1)  # Identity covariance in CLR space (D-1 x D-1)

# Create the Aitchison distribution using TransformedDistribution
aitchison_dist = Aitchison(mean, covariance_matrix)

# Sample from the Aitchison distribution
samples = aitchison_dist.sample((5,))
print("Samples from Aitchison distribution on the simplex:")
print(samples)

# Compute log probability for a sample
log_prob = aitchison_dist.log_prob(samples)
print("\nLog-probability of the samples:")
print(log_prob)

# Entropy of the Aitchison distribution
entropy = aitchison_dist.entropy()
print("\nEntropy of the Aitchison distribution:")
print(entropy)