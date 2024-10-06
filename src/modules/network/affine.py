import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Linear):
    def __init__(self, mixing_matrix=None, mixture_initialization=None, bias=True, **kwargs):
        super(Network, self).__init__(*mixing_matrix.size(), bias=bias)

        if mixing_matrix is not None:
            self.weight = nn.Parameter(mixing_matrix)

        self.mixture_initialization = mixture_initialization
        if self.mixture_initialization:
            self.on_train_start()

    def on_train_start(self):
        if self.mixture_initialization:
            getattr(nn.init, self.mixture_initialization)(self.weight)

    def eval(self):
        super(Network, self).eval()
        self.requires_grad_(False)

    @property
    def matrix(self):
        last_column = self.weight[:, -1].unsqueeze(1)
        bar_A = self.weight[:, :-1] - last_column

        # Check affine independence by checking full column rank
        # rank = torch.linalg.matrix_rank(bar_A)
        # full_rank = bar_A.size(1)  # The number of columns in bar_A (n-1)
        #
        # # Return the affine independence check result
        # affinely_independent = rank == full_rank
        #
        # # Perform the standard linear transformation on the input data
        return bar_A

    @property
    def shift(self):
        return self.weight[:, -1]

    def forward(self, data):
        return F.linear(data[..., :-1], self.matrix, self.shift)