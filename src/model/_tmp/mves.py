import torch
import numpy as np
from scipy.linalg import det, inv
from scipy.optimize import linprog
import math


class MinimumVolumeEnclosingSimplex:
    def __init__(self):
        pass

    def compute_volume(self, B):
        """Computes the volume of the simplex given matrix B."""
        # B should have dimensions (n, n+1), where n is the number of dimensions.
        if B.shape[1] != B.shape[0] + 1:
            raise ValueError("For a valid simplex, the matrix should have n dimensions and n+1 vertices.")

        # Subtract the first vertex from all others to create a reduced matrix
        B_reduced = B[:, 1:] - B[:, [0]]  # Broadcasting to subtract the first vertex from all others

        # Compute the volume using the determinant of the reduced matrix
        volume = torch.abs(torch.linalg.det(B_reduced)) / torch.tensor(math.factorial(B.shape[0]))
        return volume

    def affine_transform(self, X):
        """Performs affine transformation of the data to reduce dimensionality."""
        mean_X = torch.mean(X, dim=1, keepdim=True)  # Compute mean along axis 1
        X_centered = X - mean_X
        U, _, _ = torch.svd(X_centered)
        return torch.matmul(U[:, :X.shape[0] - 1].T, X_centered), mean_X, U

    def estimate_simplex_vertices(self, X):
        """
        Estimate the vertices of the minimum-volume enclosing simplex (MVES) for a set of vectors.
        X: A matrix where each column is a vector (M-dimensional).
        Returns:
        - simplex_vertices: A matrix where each column represents a vertex of the estimated simplex.
        """
        num_vertices = X.shape[0] + 1

        # Step 1: Affine transform of the data
        X_reduced, d, U = self.affine_transform(X)

        # Initialize the simplex vertices
        simplex_vertices = X_reduced[:, :num_vertices - 1]

        # Cyclic optimization to minimize the volume
        for i in range(num_vertices - 1):
            if simplex_vertices.shape[0] == simplex_vertices.shape[1]:
                H = inv(simplex_vertices)  # Inverse only for square matrices
                g = torch.matmul(H, simplex_vertices[:, -1])

                # Partial maximization for each row
                for j in range(H.shape[0]):
                    row_H = torch.zeros(H.shape[1])
                    row_H[j] = 1  # Set the current element

                    A_ub = X_reduced.T.cpu().numpy()  # A_ub for linprog needs to be a numpy array
                    b_ub = np.ones(A_ub.shape[0])  # b_ub should match the number of rows in A_ub

                    result = linprog(c=row_H.cpu().numpy(), A_ub=A_ub, b_ub=b_ub, method='highs')
                    if result.success:
                        simplex_vertices[:, j] = torch.tensor(result.x).to(X.device)

        # Ensure the estimated simplex has exactly n+1 vertices
        if simplex_vertices.shape[1] < num_vertices:
            # Add a random additional vertex (this is just a fallback strategy)
            extra_vertex = torch.mean(simplex_vertices, dim=1, keepdim=True)
            simplex_vertices = torch.cat([simplex_vertices, extra_vertex], dim=1)
        elif simplex_vertices.shape[1] > num_vertices:
            simplex_vertices = simplex_vertices[:, :num_vertices]  # Truncate to n+1 vertices

        # Step 2: Recover the vertices in the original space
        simplex_vertices_original = torch.matmul(U[:, :X.shape[0] - 1], simplex_vertices) + d

        return simplex_vertices_original


# Example usage
def generate_synthetic_data(n_points, n_dim, transformation_matrix):
    Z = np.random.dirichlet(alpha=[1] * (n_dim + 1), size=n_points).T  # Points on the unit simplex
    X = torch.matmul(transformation_matrix, torch.tensor(Z, dtype=torch.float32))
    return X


# Parameters
# true_vertices = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5]]).T  # 3D space, 4 vertices
# n_points = 1000
# n_dim = 3
#
# # Generate data
# X = generate_synthetic_data(n_points, n_dim, true_vertices)
#
# # Instantiate the MVES class and estimate vertices
# mves = MinimumVolumeEnclosingSimplex()
# estimated_vertices = mves.estimate_simplex_vertices(X)
#
# print("True Vertices:\n", true_vertices)
# print("Estimated Vertices:\n", estimated_vertices)
#
# # Compute the volume of the true and estimated simplices
# true_volume = mves.compute_volume(true_vertices)
# estimated_volume = mves.compute_volume(estimated_vertices)
#
# print(f"True Simplex Volume: {true_volume}")
# print(f"Estimated Simplex Volume: {estimated_volume}")