import torch
from torch.distributions import Transform, constraints


class Transformation(Transform):
    # domain = constraints.real_vector
    # codomain = constraints.simplex
    # bijective = True
    # sign = +1

    def __init__(self, p=2):
        super().__init__()
        self.p = p

    @property
    def domain(self):
        return constraints.simplex

    @property
    def codomain(self):
        return constraints.real_vector

    def __call__(self, sample):
        powered_sample = torch.abs(sample) ** self.p
        transformed_sample = powered_sample / (1 + powered_sample.sum(dim=-1, keepdim=True))
        # final_component = 1 / (1 + powered_sample.sum(dim=-1, keepdim=True))
        # transformed_sample = torch.cat([transformed_sample, final_component], dim=-1)
        return transformed_sample

    def _inverse(self, y):
        y_N_minus_1 = y[..., :-1]
        y_N = y[..., -1:]
        return y_N_minus_1 * (1 / y_N).pow(1 / self.p)

    # def _inverse(self, x):
    #     powered_x = torch.abs(x) ** self.p
    #     sum_powered = powered_x.sum(dim=-1, keepdim=True)
    #
    #     transformed_x = powered_x / (1 + sum_powered)
    #     final_component = 1 / (1 + sum_powered)
    #     return torch.cat([transformed_x, final_component], dim=-1)


    def log_abs_det_jacobian(self, x, y):
        powered_x = torch.abs(x) ** self.p
        sum_powered = powered_x.sum(dim=-1, keepdim=True)
        log_jacobian = (self.p - 1) * torch.log(torch.abs(x)).sum(dim=-1) - (self.p * torch.log(1 + sum_powered))
        return log_jacobian


# class Transformation(Transform):
#     domain = constraints.real_vector
#     codomain = constraints.simplex
#     bijective = True
#     sign = +1
#
#     def __init__(self):
#         super().__init__()
#
#     def __call__(self, sample):
#         sqrt_sample = torch.sqrt(sample)
#         norm_factor = torch.sqrt(sample.sum(dim=-1, keepdim=True))
#         return sqrt_sample * norm_factor
#
#     def _inverse(self, sample):
#         squared_sample = sample.pow(2)
#         normalized_sample = squared_sample / squared_sample.sum(dim=-1, keepdim=True)
#         return normalized_sample
#
#     def log_abs_det_jacobian(self, x, y):
#         return (-0.5 * torch.log(x)).sum(dim=-1)
