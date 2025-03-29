import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def spectral_angle_mapper(a_est, a_true):
    num = torch.sum(a_true * a_est, dim=0)
    denom = torch.norm(a_true, dim=0) * torch.norm(a_est, dim=0) + 1e-12

    angle = torch.acos((torch.clamp(num / denom, -1.0, 1.0)))
    angle = angle.mean()

    # from torchmetrics.image import SpectralAngleMapper
    # sam = SpectralAngleMapper()
    # angle = sam(a_true.unsqueeze(0).unsqueeze(3), a_est.unsqueeze(0).unsqueeze(3))

    return angle


from sklearn.cluster import KMeans
import torch


def kmeans_torch(data, num_clusters, random_state=42):
    np.random.seed(random_state)
    data_np = data.cpu().numpy()
    kmeans = KMeans(n_clusters=num_clusters, max_iter=1000)
    kmeans.fit(data_np)
    return torch.tensor(kmeans.cluster_centers_, device=data.device)


def match_components(matrix_true, matrix_est, vector_est=None):
    """
    Applies the Hungarian Algorithm to match permuted components between two vector spaces
    """
    norms = torch.norm(matrix_true, dim=0)
    norms_est = torch.norm(matrix_est, dim=0)
    num = matrix_true.T @ matrix_est
    denom = norms.unsqueeze(1) * norms_est.unsqueeze(0) + 1e-12
    cosine = num / denom
    angle = torch.acos(torch.clamp(cosine, -1.0, 1.0))

    row_ind, col_ind = linear_sum_assignment(angle.detach().cpu().numpy())
    matrix_est_matched = matrix_est[:, col_ind]

    if vector_est is not None:
        vector_est_matched = vector_est[:, col_ind]
        return matrix_est_matched, vector_est_matched

    return matrix_est_matched
