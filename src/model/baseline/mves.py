# import numpy as np
# import math
# from scipy.spatial import ConvexHull
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
#
# def simplex_volume(V):
#     """
#     Compute the volume of a simplex given its vertices.
#     The matrix V should have n+1 vertices (columns) and n dimensions (rows).
#     """
#     if V.shape[1] != V.shape[0] + 1:
#         raise ValueError("For a valid simplex, the matrix should have n dimensions and n+1 vertices.")
#
#     # Subtract the first vertex from all others (we're treating the first vertex as the "origin" for the simplex)
#     V_reduced = V[:, 1:] - V[:, [0]]  # Use broadcasting to subtract the first vertex from the rest
#
#     # Compute the volume using the determinant
#     volume = np.abs(np.linalg.det(V_reduced)) / math.factorial(V.shape[0])
#     return volume
#
#
# def generate_simplex_data(n_points, n_dim, transformation_matrix):
#     """
#     Generate data from a simplex transformation.
#     Arguments:
#     n_points -- Number of data points to generate
#     n_dim -- Dimensionality of the simplex
#     transformation_matrix -- Linear transformation matrix (true simplex)
#
#     Returns:
#     X -- Transformed data points
#     Z -- Original points on the simplex
#     """
#     # Ensure the transformation_matrix has dimensions (n_dim, n_dim + 1)
#     if transformation_matrix.shape[1] != n_dim + 1:
#         raise ValueError(f"transformation_matrix must have {n_dim + 1} columns for an {n_dim}-dimensional simplex.")
#
#     # Generate points on the unit simplex (with n_dim + 1 vertices)
#     Z = np.random.dirichlet(alpha=[1] * (n_dim + 1), size=n_points).T  # Points on the unit simplex
#
#     # Apply the linear transformation to the points
#     X = transformation_matrix @ Z  # Apply the linear transformation
#     return X, Z
#
#
# def estimate_simplex_vertices(X, n_vertices):
#     """
#     Estimate simplex vertices using Convex Hull and select the closest n_vertices.
#     X -- Data points in observed space (n_dim x n_points).
#     n_vertices -- Expected number of vertices for the simplex (n_dim + 1).
#
#     Returns:
#     A -- Estimated vertices of the simplex (n_dim x n_vertices).
#     """
#     hull = ConvexHull(X.T)
#     hull_vertices = X[:, hull.vertices]
#
#     # If the convex hull returns more vertices than expected, select the closest vertices
#     if hull_vertices.shape[1] > n_vertices:
#         # Calculate the centroid of the points
#         centroid = np.mean(X, axis=1, keepdims=True)
#
#         # Calculate distances from each vertex to the centroid
#         distances = np.linalg.norm(hull_vertices - centroid, axis=0)
#
#         # Sort by distances and select the closest n_vertices
#         selected_indices = np.argsort(distances)[:n_vertices]
#         return hull_vertices[:, selected_indices]
#     else:
#         return hull_vertices
#
#
# def plot_simplex(true_vertices, estimated_vertices, X):
#     """
#     Plot the true and estimated simplex vertices in 3D, along with the data points.
#     X is the data after transformation (in the observed space).
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     # Plot the generated data points in the transformed space (X)
#     ax.scatter(X[0], X[1], X[2], color='c', alpha=0.5, label='Data Points')
#
#     # Plot the true vertices
#     ax.scatter(true_vertices[0], true_vertices[1], true_vertices[2], color='r', label='True Vertices', s=100)
#
#     # Plot the estimated vertices
#     ax.scatter(estimated_vertices[0], estimated_vertices[1], estimated_vertices[2], color='b',
#                label='Estimated Vertices', s=100)
#
#     # Connect corresponding true and estimated vertices with green dashed lines
#     for i in range(true_vertices.shape[1]):
#         ax.plot([true_vertices[0, i], estimated_vertices[0, i]],
#                 [true_vertices[1, i], estimated_vertices[1, i]],
#                 [true_vertices[2, i], estimated_vertices[2, i]], 'g--')
#
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#
#     ax.legend()
#     plt.show()
#
#
# if __name__ == "__main__":
#     np.random.seed(42)
#
#     # 1. Set dimensions
#     n_points = 100  # Number of data points
#     n_dim = 3  # Dimensionality of the simplex
#
#     # 2. Define true simplex vertices via a known transformation matrix
#     true_vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
#     transformation_matrix = np.array([
#         [1, 0, 0, 0.5],  # 3 rows (3D space), 4 columns (4 vertices)
#         [0, 1, 0, 0.5],
#         [0, 0, 1, 0.5]
#     ])
#
#     # Generate the data
#     X, Z = generate_simplex_data(n_points, n_dim, transformation_matrix)
#
#     # Estimate the simplex vertices using Convex Hull and select the closest vertices
#     estimated_vertices = estimate_simplex_vertices(X, n_vertices=n_dim + 1)
#
#     # 5. Compare true and estimated vertices (mean squared error)
#     mse = mean_squared_error(transformation_matrix.T, estimated_vertices.T)
#     print(f"Mean Squared Error between true and estimated vertices: {mse:.6f}")
#
#     # 6. Output true and estimated simplex volumes
#     true_volume = simplex_volume(transformation_matrix)
#     estimated_volume = simplex_volume(estimated_vertices)
#     print(f"True Simplex Volume: {true_volume}")
#     print(f"Estimated Simplex Volume: {estimated_volume}")
#
#     # 7. Visualize the true and estimated vertices
#     plot_simplex(transformation_matrix, estimated_vertices, X)
#
#
#
# import numpy as np
# from scipy.optimize import linprog
# from scipy.linalg import det, inv
#
#
# def estimate_simplex_vertices(X):
#     """
#     Estimate the vertices of the minimum-volume enclosing simplex (MVES) for a set of vectors.
#
#     Parameters:
#     - X: A matrix where each column is a vector (M-dimensional).
#
#     Returns:
#     - simplex_vertices: A matrix where each column represents a vertex of the estimated simplex.
#     """
#
#     def compute_volume(B):
#         """Computes the volume of the simplex given matrix B."""
#         return np.abs(det(B)) / np.math.factorial(B.shape[1])
#
#     def affine_transform(X):
#         """Performs affine transformation of the data to reduce dimensionality."""
#         mean_X = np.mean(X, axis=1, keepdims=True)  # Compute mean along axis 1
#         X_centered = X - mean_X
#         U, _, _ = np.linalg.svd(X_centered)
#         return U[:, :X.shape[0] - 1].T @ X_centered, mean_X
#
#     # The number of vertices is equal to the dimensionality of the data + 1
#     num_vertices = X.shape[0] + 1
#
#     # Step 1: Affine transform of the data
#     X_reduced, d = affine_transform(X)
#
#     # Initialize the simplex vertices with the first 'num_vertices - 1' reduced data points
#     simplex_vertices = X_reduced[:, :num_vertices - 1]
#
#     # Cyclic optimization to minimize the volume
#     for i in range(num_vertices - 1):
#         # Ensure we are operating on a square matrix for H
#         if simplex_vertices.shape[0] == simplex_vertices.shape[1]:
#             H = inv(simplex_vertices)  # Inverse is only defined for square matrices
#             g = H @ simplex_vertices[:, -1]
#
#             # Partial maximization for each row
#             for j in range(H.shape[0]):  # Loop through the reduced dimensions
#                 row_H = np.zeros(H.shape[1])
#                 row_H[j] = 1  # Set only the current element to 1
#
#                 # Ensure A_ub and b_ub have matching dimensions
#                 A_ub = X_reduced.T
#                 b_ub = np.ones(A_ub.shape[0])  # b_ub is now 1-D with the same number of rows as A_ub
#
#                 # Use HiGHS solver
#                 result = linprog(c=row_H, A_ub=A_ub, b_ub=b_ub, method='highs')
#                 if result.success:
#                     simplex_vertices[:, j] = result.x
#
#     # Adjust the dimensionality of d for broadcasting with simplex_vertices
#     d_expanded = np.tile(d, (1, simplex_vertices.shape[1]))
#
#     # Recover the vertices in the original dimension
#     simplex_vertices = d_expanded + simplex_vertices
#
#     return simplex_vertices
#
#
# # Example usage
# X = np.random.rand(5, 100)  # 5-dimensional data, 100 samples
# vertices = estimate_simplex_vertices(X)
# print(vertices)
#
# # todo: implement mves
