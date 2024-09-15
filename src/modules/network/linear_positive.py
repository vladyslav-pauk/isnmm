import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Linear):
    def __init__(self, mixing_matrix, init_weights=None):
        super(Network, self).__init__(mixing_matrix.shape[1], mixing_matrix.shape[0], bias=False)
        self.weight = nn.Parameter(mixing_matrix)
        if init_weights:
            getattr(nn.init, init_weights)(self.weight)

    @property
    def matrix(self):
        return self.weight.abs()

    def forward(self, input):
        return F.linear(input, self.matrix, self.bias)

# fixme: verify this 21
# class LinearPositive(nn.Module):
#     def __init__(self, in_features, out_features):
#         super().__init__()
#         self.weight = nn.Parameter(torch.rand(out_features, in_features))
#         self.bias = nn.Parameter(torch.zeros(out_features))
#
#     def forward(self, x):
#         weight = F.softplus(self.weight)  # Ensure weights are positive
#         return F.linear(x, weight, self.bias)