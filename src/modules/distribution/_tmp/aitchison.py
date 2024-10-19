import torch
from torch.distributions import Distribution, MultivariateNormal
from torch.distributions import constraints


class Aitchison(Distribution):
    arg_constraints = {'mean': constraints.real, 'covariance_matrix': constraints.positive_definite}
    support = constraints.simplex

    def __init__(self, mean, covariance_matrix, validate_args=None):
        """
        Aitchison distribution that operates over the simplex, based on a multivariate
        normal distribution in centered log-ratio (CLR) space.

        Args:
            mean (Tensor): Mean vector of the distribution in CLR space (D-1 dimensional).
            covariance_matrix (Tensor): Covariance matrix in CLR space (D-1 x D-1).
        """
        # Initialize parameters
        self._mean = mean  # Use _mean to avoid conflict with the mean property
        self.covariance_matrix = covariance_matrix
        self.multivariate_normal = MultivariateNormal(mean, covariance_matrix)

        # Dimension in CLR space
        self.D_minus_1 = mean.size(-1)
        self.D = self.D_minus_1 + 1  # Full simplex dimension

        super(Aitchison, self).__init__(validate_args=validate_args)

    @property
    def mean(self):
        """Get the mean of the Aitchison distribution."""
        return self._mean

    def sample(self, sample_shape=torch.Size()):
        """Generates samples from the Aitchison distribution."""
        # Sample from multivariate normal in CLR space
        clr_sample = self.multivariate_normal.sample(sample_shape)

        # Inverse CLR to map to the simplex
        return self._inverse_clr(clr_sample)

    def log_prob(self, value):
        """
        Computes the log probability of a sample.

        Args:
            value (Tensor): A sample on the simplex.

        Returns:
            Tensor: Log-probability.
        """
        # Apply the CLR transformation
        clr_value = self._clr(value)

        # Compute log probability in the CLR space (Multivariate Normal)
        log_prob_clr = self.multivariate_normal.log_prob(clr_value)

        # Jacobian correction term: -sum(log(x_i)) for the components on the simplex
        jacobian_correction = torch.sum(torch.log(value), dim=-1)

        return log_prob_clr - jacobian_correction

    def entropy(self):
        """Computes the entropy of the Aitchison distribution."""
        entropy_clr = self.multivariate_normal.entropy()
        # Expected value of -log(x_i)
        expected_log_simplex = -torch.digamma(torch.tensor(self.D))  # Approximate using digamma function
        return entropy_clr + expected_log_simplex

    def _clr(self, x):
        """
        Centered log-ratio (CLR) transformation.
        Args:
            x (Tensor): Input composition on the simplex.
        Returns:
            Tensor: CLR transformed values in Euclidean space.
        """
        geometric_mean = torch.exp(torch.mean(torch.log(x), dim=-1, keepdim=True))
        return torch.log(x / geometric_mean)

    def _inverse_clr(self, z):
        """
        Inverse CLR transformation: maps from CLR space to the simplex.
        Args:
            z (Tensor): Input in CLR space.
        Returns:
            Tensor: Values on the simplex.
        """
        exp_z = torch.exp(z)
        return exp_z / torch.sum(exp_z, dim=-1, keepdim=True)


# Example usage
D = 3  # Number of components in the simplex
mean = torch.zeros(D - 1)  # Mean vector in CLR space (D-1 dimensional)
covariance_matrix = torch.eye(D - 1)  # Identity covariance in CLR space (D-1 x D-1)

# Create the Aitchison distribution
aitchison_dist = Aitchison(mean, covariance_matrix)

# Sample from the Aitchison distribution
samples = aitchison_dist.sample((5,))
print("Samples:", samples)

# Compute log probability for a sample
log_prob = aitchison_dist.log_prob(samples)
print("Log-Probability:", log_prob)

# Compute entropy
entropy = aitchison_dist.entropy()
print("Entropy:", entropy)