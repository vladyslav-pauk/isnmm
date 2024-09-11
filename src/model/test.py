import numpy as np
from scipy.optimize import linprog
from scipy.linalg import det, inv


def estimate_simplex_vertices(X):
    """
    Estimate the vertices of the minimum-volume enclosing simplex (MVES) for a set of vectors.

    Parameters:
    - X: A matrix where each column is a vector (M-dimensional).

    Returns:
    - simplex_vertices: A matrix where each column represents a vertex of the estimated simplex.
    """

    def compute_volume(B):
        """Computes the volume of the simplex given matrix B."""
        return np.abs(det(B)) / np.math.factorial(B.shape[1])

    def affine_transform(X):
        """Performs affine transformation of the data to reduce dimensionality."""
        mean_X = np.mean(X, axis=1, keepdims=True)  # Compute mean along axis 1
        X_centered = X - mean_X
        U, _, _ = np.linalg.svd(X_centered)
        return U[:, :X.shape[0] - 1].T @ X_centered, mean_X

    # The number of vertices is equal to the dimensionality of the data + 1
    num_vertices = X.shape[0] + 1

    # Step 1: Affine transform of the data
    X_reduced, d = affine_transform(X)

    # Initialize the simplex vertices with the first 'num_vertices - 1' reduced data points
    simplex_vertices = X_reduced[:, :num_vertices - 1]

    # Cyclic optimization to minimize the volume
    for i in range(num_vertices - 1):
        # Ensure we are operating on a square matrix for H
        if simplex_vertices.shape[0] == simplex_vertices.shape[1]:
            H = inv(simplex_vertices)  # Inverse is only defined for square matrices
            g = H @ simplex_vertices[:, -1]

            # Partial maximization for each row
            for j in range(H.shape[0]):  # Loop through the reduced dimensions
                row_H = np.zeros(H.shape[1])
                row_H[j] = 1  # Set only the current element to 1

                # Ensure A_ub and b_ub have matching dimensions
                A_ub = X_reduced.T
                b_ub = np.ones(A_ub.shape[0])  # b_ub is now 1-D with the same number of rows as A_ub

                # Use HiGHS solver
                result = linprog(c=row_H, A_ub=A_ub, b_ub=b_ub, method='highs')
                if result.success:
                    simplex_vertices[:, j] = result.x

    # Adjust the dimensionality of d for broadcasting with simplex_vertices
    d_expanded = np.tile(d, (1, simplex_vertices.shape[1]))

    # Recover the vertices in the original dimension
    simplex_vertices = d_expanded + simplex_vertices

    return simplex_vertices


# Example usage
X = np.random.rand(5, 100)  # 5-dimensional data, 100 samples
vertices = estimate_simplex_vertices(X)
print(vertices)