from numpy import random
import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, latent_dim, observed_dim, degree, nonlinearity, init_weights=None):
        super(Network, self).__init__()
        self.model = nonlinearity
        self.degree = degree if degree is not None else 0
        self.coefficients = torch.tensor(random.rand(observed_dim, self.degree + 1))
        if init_weights:
            getattr(nn.init, init_weights, lambda x: x)(self.coefficients)

    def forward(self, x):
        transformed_components = [
            self.component_transform(x[..., i:i + 1], self.coefficients[i]) for i in range(x.shape[-1])
        ]
        return torch.cat(transformed_components, dim=-1)

    def component_transform(self, x, coefficients):
        if self.model == None:
            return x

        transformed_components = torch.stack(
            [self.basis_function(x, power, coeff) for power, coeff in enumerate(coefficients)]
        ).sum(dim=0)
        return transformed_components

    def basis_function(self, x, power, coeff):
        # x = 10 * torch.tanh(0.5 * x ** 2) + x + x**2
        if self.model == 'tanh':
            x = coeff * torch.tanh((power + 1) * x ** power)
        elif self.model == 'sin':
            x = torch.sin(4 * x) + 5 * x
        return x

    def inverse(self, y, tol=1e-6, max_iter=10000):
        x = y.clone()

        for _ in range(max_iter):
            f_x = self.forward(x)
            diff = y - f_x
            if torch.norm(diff) < tol:
                break

            x = x + diff * 0.1
        return x

    # def inverse(self, y, num_iterations=1000):
    #     """ Numerically approximate the inverse transformation using Newton's method """
    #     x_approx = y.clone()
    #     for _ in range(num_iterations):
    #         f_x = self.forward(x_approx) - y
    #         f_x_prime = self.derivative(x_approx)
    #         x_approx -= f_x / (f_x_prime + 1e-6)
    #     return x_approx
    # def inverse(self, y, **kwargs):
    #     # y = (torch.atanh(y / 10) / 0.5).pow(1 / 2)
    #     y = torch.log(2 * y)
    #     return y


    # todo: use the same transform as in the network, just randomize weights? proof of principle, then use arbitrary
    #  nonlinearity
    # todo: make it distributions.transform class? it has inv and derivative
