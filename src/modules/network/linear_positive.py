import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Linear):
    def __init__(self, mixing_matrix, mixture_initialization=None, **kwargs):
        super(Network, self).__init__(*mixing_matrix.size(), bias=False)
        self.weight = nn.Parameter(mixing_matrix)
        self.mixture_initialization = mixture_initialization

    def on_train_start(self):
        getattr(nn.init, self.mixture_initialization)(self.weight) if self.mixture_initialization else lambda x: x

    def eval(self):
        super(Network, self).eval()
        self.requires_grad_(False)

    @property
    def matrix(self):
        return self.weight.abs()

    def forward(self, data):
        return F.linear(data, self.matrix, None)
