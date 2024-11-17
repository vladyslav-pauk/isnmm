import torch
from scipy.optimize import linear_sum_assignment


def spectral_angle_mapper(a_est, a_true):
    num = torch.sum(a_true * a_est, dim=0)
    denom = torch.norm(a_true, dim=0) * torch.norm(a_est, dim=0)

    angle = torch.acos((torch.clamp(num / denom, -1.0, 1.0)))
    angle = angle.mean()

    # from torchmetrics.image import SpectralAngleMapper
    # sam = SpectralAngleMapper()
    # angle = sam(a_true.unsqueeze(0).unsqueeze(3), a_est.unsqueeze(0).unsqueeze(3))

    return angle


def kmeans_torch(data, num_clusters, num_iters=10):
    """
    Performs KMeans clustering using PyTorch.

    Parameters:
    data (torch.Tensor): Data tensor of shape (num_samples, num_features).
    num_clusters (int): Number of clusters.
    num_iters (int): Number of iterations.

    Returns:
    torch.Tensor: Cluster centers of shape (num_clusters, num_features).
    """
    num_samples, dim = data.shape
    indices = torch.randperm(num_samples)[:num_clusters]
    centers = data[indices].clone()

    for _ in range(num_iters):
        distances = torch.cdist(data, centers)
        labels = torch.argmin(distances, dim=1)
        for k in range(num_clusters):
            if torch.sum(labels == k) == 0:
                centers[k] = data[torch.randint(0, num_samples, (1,))]
            else:
                centers[k] = data[labels == k].mean(dim=0)
    return centers


def match_components(matrix_true, matrix_est, vector_est=None):
    """
    Applies the Hungarian Algorithm to match permuted components between two vector spaces
    """
    norms = torch.norm(matrix_true, dim=0)
    norms_est = torch.norm(matrix_est, dim=0)
    num = matrix_true.T @ matrix_est
    denom = norms.unsqueeze(1) * norms_est.unsqueeze(0)
    cosine = num / denom
    angle = torch.acos(torch.clamp(cosine, -1.0, 1.0))
    row_ind, col_ind = linear_sum_assignment(angle.cpu().numpy())

    matrix_est_matched = matrix_est[:, col_ind]

    vector_est_matched = None
    if vector_est is not None:
        vector_est_matched = vector_est[:, col_ind]
        return matrix_est_matched, vector_est_matched
    return matrix_est_matched
