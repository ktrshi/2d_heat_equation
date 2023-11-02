import numpy as np
from matplotlib import gridspec
from scipy.sparse import lil_matrix, diags, identity
from scipy.sparse.linalg import spsolve, splu
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm

# Параметры сетки и времени
N = M = 201
dx, dy = 1.0 / (N - 1), 1.0 / (M - 1)
dt = 0.0005
Dx, Dy = dt / dx ** 2, dt / dy ** 2
D = dt / dx ** 2
D_f = dt / (2 * dx ** 2)
Tmax = 4
Nt = int(Tmax / dt)
x = np.linspace(0, 1, N, dtype=np.float64)
y = np.linspace(0, 1, M, dtype=np.float64)
X, Y = np.meshgrid(x, y, indexing='ij')

U_numerical = {'5pt': [], '5pt+t': [], '9pt': [], '9pt+t': []}
U_exact = []
U_diff = {'5pt': [], '5pt+t': [], '9pt': [], '9pt+t': []}


def compute_f_vectorized(u_current):
    global N, M

    RHS = np.zeros_like(u_current, dtype=np.float64)

    RHS[1:-1, 1:-1] = (1 - 2 * Dx) * u_current[1:-1, 1:-1] + \
                      Dx / 2 * (u_current[2:, 1:-1] + u_current[:-2, 1:-1]) + \
                      Dy / 2 * (u_current[1:-1, 2:] + u_current[1:-1, :-2])

    RHS[0, 1:-1] = (1 - 2 * Dx) * u_current[0, 1:-1] + \
                   Dx  * u_current[1, 1:-1] + \
                   Dy / 2 * (u_current[0, 2:] + u_current[0, :-2])

    RHS[1:-1, 0] = (1 - 2 * Dx) * u_current[1:-1, 0] + \
                   Dx / 2 * (u_current[2:, 0] + u_current[:-2, 0]) + \
                   Dy * u_current[1:-1, 1]

    RHS[0, 0] = (1 - 2 * Dx) * u_current[0, 0] + \
                Dx * u_current[1, 0] + \
                Dy * u_current[0, 1]

    RHS[-1, :] = 0
    RHS[:, -1] = 0

    return RHS.flatten()


def create_solution_graphs(key, tidx):
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
                im = ax.imshow(U_numerical[key][tidx], origin='lower', cmap='plasma', extent=[0, 1, 0, 1])
                ax.set_title('Численное решение')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.cax.colorbar(im)
                ax.cax.toggle_label(True)
            case 1:
                im = ax.imshow(U_exact[tidx], origin='lower', cmap='plasma', extent=[0, 1, 0, 1])
                ax.set_title('Аналитическое решение')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.cax.colorbar(im)
                ax.cax.toggle_label(True)
            case 2:
                im = ax.imshow(U_diff[key][tidx], origin='lower', cmap='plasma', extent=[0, 1, 0, 1])
                ax.set_title('Разность решений')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.cax.colorbar(im)
                ax.cax.toggle_label(True)
    plt.tight_layout()
    plt.savefig(f'unsteady_{key}_{tidx}')


def analytical_solution(t):
    global X, Y
    result = np.exp(-np.pi ** 2 / 2 * t) * np.sin(np.pi / 2 * (1 - X)) * np.sin(np.pi / 2 * (1 - Y))
    return result


if __name__ == '__main__':
    # Начальное условие
    U = {
        '5pt': np.sin(np.pi / 2 * (1 - X)) * np.sin(np.pi / 2 * (1 - Y)),
        '5pt+t': np.sin(np.pi / 2 * (1 - X)) * np.sin(np.pi / 2 * (1 - Y)),
        '9pt': np.sin(np.pi / 2 * (1 - X)) * np.sin(np.pi / 2 * (1 - Y)),
        '9pt+t': np.sin(np.pi / 2 * (1 - X)) * np.sin(np.pi / 2 * (1 - Y))
    }
    # Конструкция матрицы A для системы A*U_next = U_current
    A = {
        '5pt': lil_matrix((N * M, N * M), dtype=np.float64),
        '5pt+t': lil_matrix((N * M, N * M), dtype=np.float64),
        '9pt': lil_matrix((N * M, N * M), dtype=np.float64),
        '9pt+t': lil_matrix((N * M, N * M), dtype=np.float64)
    }
    B = lil_matrix((N * M, N * M), dtype=np.float64)
    F = {
        '5pt': np.zeros(N * M, dtype=np.float64),
        '5pt+t': np.zeros(N * M, dtype=np.float64),
        '9pt': np.zeros(N * M, dtype=np.float64),
        '9pt+t': np.zeros(N * M, dtype=np.float64)
    }
    U_1D = {
        '5pt': None,
        '5pt+t': None,
        '9pt': None,
        '9pt+t': None
    }
    LU = {'5pt': None, '5pt+t': None, '9pt': None, '9pt+t': None}

    for i in range(N):
        for j in range(M):
            idx = i + j * N

            if i == N - 1 or j == M - 1:  # Правая и верхняя границы (условие Дирихле)
                A['5pt'][idx, idx] = 1
                continue

            if i == 0 and j == 0:  # Угловая точка (0, 0)
                A['5pt'][idx, idx] = 1 + 2 * (Dx + Dy)
                A['5pt'][idx, idx + 1] = -2 * Dx
                A['5pt'][idx, idx + N] = -2 * Dy
                continue

            if i == 0:  # Левая граница
                A['5pt'][idx, idx] = 1 + 2 * (Dx + Dy)
                A['5pt'][idx, idx + 1] = -2 * Dx
                if j > 0:
                    A['5pt'][idx, idx - N] = -Dy
                if j < N - 1:
                    A['5pt'][idx, idx + N] = -Dy

            elif j == 0:  # Нижняя граница
                A['5pt'][idx, idx] = 1 + 2 * (Dx + Dy)
                A['5pt'][idx, idx + N] = -2 * Dy
                if i > 0:
                    A['5pt'][idx, idx - 1] = -Dx
                if i < N - 1:
                    A['5pt'][idx, idx + 1] = -Dx

            else:  # Внутренние узлы
                A['5pt'][idx, idx] = 1 + 2 * (Dx + Dy)
                A['5pt'][idx, idx - 1] = -Dx
                A['5pt'][idx, idx + 1] = -Dx
                A['5pt'][idx, idx - N] = -Dy
                A['5pt'][idx, idx + N] = -Dy

    # for i in range(N):
    #     for j in range(M):
    #         idx = i + j * N
    #         A['5pt+t'][idx, idx] = 1 + 4 * Dx
    #
    #         # Dirichlet boundary conditions for top and right borders
    #         if i == N - 1 or j == M - 1:
    #             A['5pt+t'][idx, :] = 0
    #             A['5pt+t'][idx, idx] = 1
    #             continue
    #
    #         # Neumann boundary conditions for the left border
    #         if i == 0:
    #             # A['5pt+t'][idx, idx] -= Dx
    #             A['5pt+t'][idx, idx] = 1 + 3 * Dx
    #             # if idx + 1 < N * M:
    #                 # A['5pt+t'][idx, idx + 1] = -Dx
    #
    #         # Neumann boundary conditions for the bottom border
    #         if j == 0:
    #             A['5pt+t'][idx, idx] -= 1 + 3 * Dx
    #             # if idx + N < N * M:
    #                 # A['5pt+t'][idx, idx + N] = -Dy
    #
    #         # For interior points and points on Neumann boundary
    #         if i > 0 and idx - 1 >= 0:
    #             A['5pt+t'][idx, idx - 1] = -Dx / 2
    #         if i < N - 1 and idx + 1 < N * M:
    #             A['5pt+t'][idx, idx + 1] = -Dx / 2
    #         if j > 0 and idx - N >= 0:
    #             A['5pt+t'][idx, idx - N] = -Dy / 2
    #         if j < M - 1 and idx + N < N * M:
    #             A['5pt+t'][idx, idx + N] = -Dy / 2

    for i in range(N):
        for j in range(N):
            idx = i * N + j

            # Внутренние узлы
            if 1 <= i < N - 1 and 1 <= j < N - 1:
                A[idx, idx] = 1 + 2 * Dx
                A[idx, idx + 1] = A[idx, idx - 1] = - Dx / 2
                A[idx, idx + N] = A[idx, idx - N] = - Dx / 2

            # Левая граница (условие Неймана)
            elif i == 0:
                A[idx, idx] = 1 + 1.5 * Dx
                A[idx, idx + 1] = -1 * Dx
                if 1 <= j < N - 1:
                    A[idx, idx + N] = A[idx, idx - N] = - Dx / 2

            # Нижняя граница (условие Неймана)
            elif j == 0:
                A[idx, idx] = 1 + 1.5 * Dx
                A[idx, idx + N] = -1 * Dx
                if 1 <= i < N - 1:
                    A[idx, idx + 1] = A[idx, idx - 1] = - Dx / 2

            # Верхняя и правая границы (условие Дирихле)
            else:
                A[idx, idx] = 1

    coeff = 1 / (6 * dx ** 2)
    for i in range(N):
        for j in range(M):
            index = i + j * N

            # Dirichlet boundary conditions on the top and right boundary
            if i == N - 1 or j == N - 1:
                A['9pt'][index, index] = 1
                continue

            A['9pt'][index, index] = 1 + 4 * D

            # Neumann boundary conditions on the left and bottom boundary
            if i == 0:
                A['9pt'][index, index] = 1 + 3 * D
            if j == 0:
                A['9pt'][index, index] = 1 + 3 * D

            # Neighbors
            if i > 0:
                A['9pt'][index, index - 1] = -D
            if i < N - 1:
                A['9pt'][index, index + 1] = -D
            if j > 0:
                A['9pt'][index, index - N] = -D
            if j < N - 1:
                A['9pt'][index, index + N] = -D

            # Diagonal neighbors
            if i > 0 and j > 0:
                A['9pt'][index, index - N - 1] = -0.25 * D
            if i < N - 1 and j > 0:
                A['9pt'][index, index - N + 1] = -0.25 * D
            if i > 0 and j < N - 1:
                A['9pt'][index, index + N - 1] = -0.25 * D
            if i < N - 1 and j < N - 1:
                A['9pt'][index, index + N + 1] = -0.25 * D

    for j in range(M):
        for i in range(N):
            index = i + j * N

            # Dirichlet boundary conditions on the top and right boundary
            if i == N - 1 or j == N - 1:
                A['9pt+t'][index, index] = 1
                B[index, index] = 1
                continue

            # Center
            A['9pt+t'][index, index] = 1 + 4 * D
            B[index, index] = 1 - 4 * D

            # Neumann boundary conditions on the left and bottom boundary
            if i == 0:
                A['9pt+t'][index, index] = 1 + 3 * D
                B[index, index] = 1 - 3 * D
            if j == 0:
                A['9pt+t'][index, index] = 1 + 3 * D
                B[index, index] = 1 - 3 * D

            # Neighbors
            neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            for dx, dy in neighbors:
                if 0 <= i + dx < N and 0 <= j + dy < N:
                    A['9pt+t'][index, index + dx + dy * N] = -D
                    B[index, index + dx + dy * N] = D

            # Diagonal neighbors
            diag_neighbors = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
            for dx, dy in diag_neighbors:
                if 0 <= i + dx < N and 0 <= j + dy < N:
                    A['9pt+t'][index, index + dx + dy * N] = -0.25 * D
                    B[index, index + dx + dy * N] = 0.25 * D

    F['5pt+t'] = compute_f_vectorized(U['5pt+t'])
    for key in ['5pt', '5pt+t', '9pt', '9pt+t']:
        LU[key] = splu(A['5pt'].tocsc())

    # for sol, key in enumerate(['5pt', '5pt+t', '9pt', '9pt+t']):
    # for sol, key in enumerate(['5pt+t', '9pt+t']):
    for sol, key in enumerate(['5pt+t']):
        for t in tqdm(range(Nt)):
            # Применение условий Дирихле
            U[key][:, -1] = 0
            U[key][-1, :] = 0
            if key in ['5pt', '9pt']:
                U_1D[key] = U[key].ravel()
                U_1D[key] = LU[key].solve(U_1D[key])
            if key in ['5pt+t']:
                F[key] = compute_f_vectorized(U[key])
                U_1D[key] = LU[key].solve(F[key])
            if key in ['9pt+t']:
                RHS = B.dot(U[key].ravel())
                U_1D[key] = LU[key].solve(RHS)
            U[key] = U_1D[key].reshape(N, M)
            U_numerical[key].append(U[key])
            if sol == 0:
                U_exact.append(analytical_solution((t+1) * dt))
            U_diff[key].append(U_exact[t] - U_numerical[key][t])
        for i in range(1, Nt, int(Nt / (5 - 1))):
            i -= 1
            create_solution_graphs(key, i)
