import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, input_dim, output_dim, nonlinearity='tanh', init_weights=None, **kwargs):
        super(Network, self).__init__()

        self.nonlinearity = nonlinearity
        layers = []
        hidden_layers = [32, 32]  # Hidden layer dimensions
        # Construct the hidden fully connected layers
        previous_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.Softplus())  # Ensuring positive weights via softplus
            previous_dim = hidden_dim

        # Output layer, mapping the last hidden layer to the desired output dimension
        layers.append(nn.Linear(previous_dim, output_dim))
        layers.append(nn.Softplus())  # Ensuring positive weights for the output layer

        self.network = nn.Sequential(*layers)
        if init_weights:
            self.apply(init_weights)

    def forward(self, x):
        return self.network(x)

    def inverse(self, y, tol=1e-6, max_iter=10000):
        # Numerically approximate inverse transformation using fixed-point iteration
        x = torch.randn(y.shape[0], self.network[0].in_features, device=y.device)  # Initialize with random tensor
        for _ in range(max_iter):
            f_x = self.forward(x)
            diff = y - f_x
            if torch.norm(diff) < tol:
                break
            x = x + diff * 0.1  # A small step size for convergence
        return x