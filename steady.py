from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu, bicgstab
import numpy as np
import matplotlib.pyplot as plt

N = M = 16
h = 1.0 / (N - 1)
A = {'5pt': lil_matrix((N * M, N * M)), '9pt': lil_matrix((N * M, N * M))}
X = {'5pt': None, '9pt': None}
X_ = {'5pt': None, '9pt': None}
LU = {'5pt': None, '9pt': None}
keys = ['5pt', '9pt']
Y = np.zeros(N * M).reshape(N, M)

for i in range(N):
    for j in range(M):
        idx = i + j * N

        A['5pt'][idx, idx] = 4

        if i == N - 1 or j == M - 1:  # Правая и верхняя границы (условие Дирихле)
            A['5pt'][idx, idx] = 1
            continue

        if i == 0 and j == 0:  # Угловая точка (0, 0)
            A['5pt'][idx, idx] = 1
            continue

        if i == 0:  # Левая граница
            A['5pt'][idx, idx] = 1


        elif j == 0:  # Нижняя граница
            A['5pt'][idx, idx] = 1

        else:  # Внутренние узлы
            A['5pt'][idx, idx - 1] = A['5pt'][idx, idx + 1] = A['5pt'][idx, idx - N] = A['5pt'][idx, idx + N] = -1

for i in range(N):
    for j in range(M):
        idx = i + j * N

        A['9pt'][idx, idx] = 8 / 3

        if i == 0 and j == 0:
            A['5pt'][idx, idx] = 1
            continue

        if i == N - 1 or j == M - 1:  # Правая и верхняя границы (условие Дирихле)
            A['9pt'][idx, idx] = 1

        # Левая граница (условие Неймана)
        elif i == 0:
            A['5pt'][idx, idx] = 1

        # Нижняя граница (условие Неймана)
        elif j == 0:
            A['5pt'][idx, idx] = 1

        # Верхняя и правая границы (условие Дирихле)
        else:
            # ближайшие соседи
            A['9pt'][idx, idx + 1] = A['9pt'][idx, idx - 1] = A['9pt'][idx, idx + N] = A['9pt'][
                idx, idx - N] = - 1 / 3

            # диагональные элементы
            A['9pt'][idx, idx + N + 1] = A['9pt'][idx, idx - N - 1] = A['9pt'][idx, idx + N - 1] = A['9pt'][
                idx, idx - N + 1] = - 1 / 3

Y[-1, :] = 1

for key in keys:
    LU[key] = splu(A[key].tocsc())
    X[key] = LU[key].solve(Y.flatten()).reshape(N, M)
    X_[key], exitcode = bicgstab(A[key].tocsc(), Y.flatten(), tol=1e-9)
difference = np.abs(X['5pt'] - X['9pt'])
difference_norm = np.linalg.norm(X['5pt'] - X['9pt'])

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
            im = ax.imshow(X['5pt'], origin='lower', cmap='plasma', extent=[0, 1, 0, 1])
            ax.set_title('Численное решение 5-ти точечный шаблон')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.cax.colorbar(im)
            ax.cax.toggle_label(True)
        case 1:
            im = ax.imshow(X['9pt'], origin='lower', cmap='plasma', extent=[0, 1, 0, 1])
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
# print(solution)
