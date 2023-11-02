from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
import matplotlib.pyplot as plt

N = M = 202
h = 1.0 / (N - 1)
A_sparse = lil_matrix((N * M, N * M))
Y = np.zeros(N * M)


for i in range(N):
    for j in range(M):
        k = i * N + j

        # Center point (main diagonal)
        A_sparse[k, k] = 4 / h ** 2

        if i > 0:  # left
            A_sparse[k, k - N] = -1 / h ** 2
        if i < N - 1:  # right
            A_sparse[k, k + N] = -1 / h ** 2
        if j > 0:  # below
            A_sparse[k, k - 1] = -1 / h ** 2
        if j < M - 1:  # above
            A_sparse[k, k + 1] = -1 / h ** 2

# Apply Dirichlet boundary conditions on the top border (U=1)
for j in range(M):
    k = (N - 1) * N + j
    A_sparse[k, :] = 0
    A_sparse[k, k] = 1
    Y[k] = 1

# Apply Dirichlet boundary conditions for U=0 on the other borders
# Left border
for i in range(1, N - 1):
    k = i * N
    A_sparse[k, :] = 0
    A_sparse[k, k] = 1
    Y[k] = 0

# Right border
for i in range(1, N - 1):
    k = i * N + (M - 1)
    A_sparse[k, :] = 0
    A_sparse[k, k] = 1
    Y[k] = 0

# Bottom border
for j in range(1, M - 1):
    k = j
    A_sparse[k, :] = 0
    A_sparse[k, k] = 1
    Y[k] = 0


# Initialize the sparse A matrix again
A_sparse_9pt = lil_matrix((N * M, N * M))

# Fill in the A matrix using 9-point stencil
for i in range(1, N - 1):
    for j in range(1, M - 1):
        k = i * N + j
        A_sparse_9pt[k, k] = 20 / 6 / h ** 2  # main diagonal
        A_sparse_9pt[k, k - 1] = A_sparse_9pt[k, k + 1] = A_sparse_9pt[k, k - N] = A_sparse_9pt[
            k, k + N] = -4 / 6 / h ** 2  # direct neighbors
        A_sparse_9pt[k, k - N - 1] = A_sparse_9pt[k, k - N + 1] = A_sparse_9pt[k, k + N - 1] = A_sparse_9pt[
            k, k + N + 1] = -1 / 6 / h ** 2  # diagonal neighbors

# Apply Dirichlet boundary conditions as before
# Top boundary
for j in range(M):
    k = j
    A_sparse_9pt[k, :] = 0
    A_sparse_9pt[k, k] = 1

# Bottom boundary
for j in range(M):
    k = (N - 1) * N + j
    A_sparse_9pt[k, :] = 0
    A_sparse_9pt[k, k] = 1

# Left boundary
for i in range(1, N - 1):
    k = i * N
    A_sparse_9pt[k, :] = 0
    A_sparse_9pt[k, k] = 1

# Right boundary
for i in range(1, N - 1):
    k = i * N + (M - 1)
    A_sparse_9pt[k, :] = 0
    A_sparse_9pt[k, k] = 1

A_sparse_9pt = A_sparse_9pt.tocsr()
A_sparse = A_sparse.tocsr()
print(not (A_sparse - A_sparse_9pt).nnz == 0)
X = spsolve(A_sparse, Y)
solution = X.reshape(N, M)
X_9pt = spsolve(A_sparse_9pt, Y)
solution_9pt = X_9pt.reshape(N, M)
difference = np.abs(solution - solution_9pt)
difference_norm = np.linalg.norm(solution - solution_9pt)
fig = plt.figure(figsize=(15, 5))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(1, 3),
                 axes_pad=0.6,
                 share_all=True,
                 label_mode='L',
                 cbar_location="right",
                 cbar_mode="each",
                 cbar_size="7%",
                 cbar_pad=0.2,
                 )

for i, ax in enumerate(grid):
    match i:
        case 0:
            im = ax.imshow(solution, origin='lower', cmap='plasma', extent=[0, 1, 0, 1])
            ax.set_title('Численное решение 5-ти точечный шаблон')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.cax.colorbar(im)
            ax.cax.toggle_label(True)
        case 1:
            im = ax.imshow(solution_9pt, origin='lower', cmap='plasma', extent=[0, 1, 0, 1])
            ax.set_title('Численное решение 9-ти точечный шаблон')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.cax.colorbar(im)
            ax.cax.toggle_label(True)
        case 2:
            im = ax.imshow(difference, origin='lower', cmap='plasma', extent=[0, 1, 0, 1])
            ax.set_title('Разность решений')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.cax.colorbar(im)
            ax.cax.toggle_label(True)
plt.tight_layout()
plt.savefig(f'steady')
print(solution)
