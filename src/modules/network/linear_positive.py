import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Linear):
    def __init__(self, mixing_matrix, mixture_initialization=None, **kwargs):
        super(Network, self).__init__(mixing_matrix.shape[1], mixing_matrix.shape[0], bias=False)
        self.weight = nn.Parameter(mixing_matrix)
        getattr(nn.init, mixture_initialization, lambda x: x)(self.weight)

    @property
    def matrix(self):
        return F.softplus(self.weight)
        # return self.weight.abs()

    def forward(self, data):
        return F.linear(data, self.matrix)
