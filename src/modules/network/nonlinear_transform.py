from numpy import random
import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, observed_dim, degree, init_weights=None):
        super(Network, self).__init__()
        self.degree = degree
        self.coefficients = torch.tensor(random.rand(observed_dim, degree + 1))
        if init_weights:
            getattr(nn.init, init_weights, lambda x: x)(self.coefficients)

    def forward(self, x):
        """ Apply the transformation component-wise: tanh(x), tanh(2x), ..., tanh(nx) """
        transformed_components = [
            torch.stack([coeff * torch.tanh((power + 1) * x[..., i:i + 1]**power) for power, coeff in enumerate(self.coefficients[i])]).sum(dim=0)
            for i in range(x.shape[-1])
        ]
        return torch.cat(transformed_components, dim=-1)

    # def inverse(self, y, num_iterations=1000):
    #     """ Numerically approximate the inverse transformation using Newton's method """
    #     x_approx = y.clone()
    #     for _ in range(num_iterations):
    #         f_x = self.forward(x_approx) - y
    #         f_x_prime = self.derivative(x_approx)
    #         x_approx -= f_x / (f_x_prime + 1e-6)
    #     return x_approx

    def inverse(self, y, tol=1e-6, max_iter=10000):
        # Initialize x with y as an approximation (you can improve this)
        x = y.clone()

        for _ in range(max_iter):
            f_x = self.forward(x)
            diff = y - f_x
            if torch.norm(diff) < tol:
                break

            # Update x using a numerical method, like Newton's method
            # Here you would compute the Jacobian or use a simpler gradient-based update
            # For simplicity, we just do a fixed-step gradient descent update
            x = x + diff * 0.1  # Step size 0.1; could be adaptive

        return x


    # todo: use the same transform as in the network, just randomize weights? proof of principle, then use arbitrary
    #  nonlinearity
    # todo: make it distributions.transform class? it has inv and derivative
