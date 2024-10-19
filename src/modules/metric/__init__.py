from .matrix_mse import MatrixMse as MatrixMse
from .residual_nonlinearity import ResidualNonlinearity as ResidualNonlinearity
from .spectral_angle import SpectralAngle as SpectralAngle
from .subspace_distance import SubspaceDistance as SubspaceDistance
from .matrix_volume import MatrixVolume as MatrixVolume
from .matrix_change import MatrixChange as MatrixChange
from .constraint_error import ConstraintError as ConstraintError

# task: check these
# def mse_matrix_db(A0, A_hat):
#     min_mse = torch.tensor(float('inf'))
#     perms = itertools.permutations(range(A0.shape[0]))
#     for perm in perms:
#         A_hat_permuted = A_hat[list(perm), :]
#         mse = torch.mean(torch.sum((A0 - A_hat_permuted) ** 2, dim=1))
#         if mse < min_mse:
#             min_mse = mse
#
#     mse_dB = 10 * torch.log10(min_mse)
#     return mse_dB
#
#
# def spectral_angle_distance(A0, A_hat):
#     A0 = A0 / torch.norm(A0, dim=1, keepdim=True)
#     A_hat = A_hat / torch.norm(A_hat, dim=1, keepdim=True)
#     cosines = torch.sum(A0 * A_hat, dim=1)
#     return torch.acos(cosines).mean()
#
#
# def subspace_distance(S, U):
#     import torch
#
#     S_pseudo_inv = torch.linalg.pinv(S)
#
#     I = torch.eye(S.shape[-1], device=S.device)
#     P_s_orth = I - S_pseudo_inv @ S
#
#     U_u, Q, V_u = torch.linalg.svd(U.T, full_matrices=False)
#     Q_u = V_u.T
#
#     matrix_product = Q_u @ P_s_orth
#
#     singular_values = torch.linalg.svd(matrix_product)[1]
#
#     norm_2 = torch.max(singular_values)
#     return norm_2