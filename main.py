import numpy as np

# Given parameters
N = M = 202
h = 1.0 / (N - 1)

# Initialize matrix A and vector Y
A = np.zeros((N * M, N * M))
Y = np.zeros(N * M)

# Populate matrix A based on the finite difference scheme
for i in range(N):
    for j in range(M):
        k = i * N + j

        # Center point (main diagonal)
        A[k, k] = -4 / h ** 2

        # Off-diagonals (neighboring points)
        if i > 0:  # left
            A[k, k - N] = 1 / h ** 2
        if i < N - 1:  # right
            A[k, k + N] = 1 / h ** 2
        if j > 0:  # below
            A[k, k - 1] = 1 / h ** 2
        if j < M - 1:  # above
            A[k, k + 1] = 1 / h ** 2

# Apply Dirichlet boundary conditions on the top border (U=1)
for j in range(M):
    k = (N - 1) * N + j
    A[k, :] = 0
    A[k, k] = 1
    Y[k] = 1

# Apply Dirichlet boundary conditions for U=0 on the other borders
# Left border
for i in range(1, N - 1):
    k = i * N
    A[k, :] = 0
    A[k, k] = 1
    Y[k] = 0

# Right border
for i in range(1, N - 1):
    k = i * N + (M - 1)
    A[k, :] = 0
    A[k, k] = 1
    Y[k] = 0

# Bottom border
for j in range(1, M - 1):
    k = j
    A[k, :] = 0
    A[k, k] = 1
    Y[k] = 0

print(A[:10, :10], Y[:10])  # Displaying the top-left corner of the matrix A and the beginning of vector Y for verification
