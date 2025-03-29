from numpy import random
import torch
import torch.nn as nn
# task: make it distributions.transform class


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
            self._component_transform(x[..., i:i + 1], i) for i in range(x.shape[-1])
        ]
        return torch.cat(transformed_components, dim=-1)

    def _component_transform(self, x, coefficient):
        if self.model == None:
            return x
        # transformed_components = torch.stack(
        #     [self._basis_function(x, power, coeff) for power, coeff in enumerate(coefficients)]
        # ).sum(dim=0)
        transformed_components = self._basis_function(x, coefficient)
        return transformed_components

    def _basis_function(self, x, coefficient):
        if self.model == 'tanh':
            # x = 10 * torch.tanh(0.5 * x ** 2) + x + x**2
            x = coefficient * torch.tanh( x ** coefficient)
        elif self.model == 'sin':
            x = torch.sin(4 * x) + 5 * x
        elif self.model == 'cnae':
            # x -= 0.25
            # x *= 20
            x = 5 * torch.sigmoid(x) + 0.3 * x
            # func = lambda x: [
            #     5 * torch.sigmoid(x) + 0.3 * x for _ in range(x.shape[-1])
                # -3 * torch.tanh(x) - 0.2 * x,
                # 0.4 * torch.exp(x)
            # ]
            # x = func(x)[coefficient]
        elif self.model == 'linear':
            x = x
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

