import numpy as np
from matplotlib import gridspec
from scipy.sparse import lil_matrix, diags, identity
import concurrent.futures
from scipy.sparse.linalg import spsolve, splu, spilu, bicgstab
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm

# Параметры сетки и времени
N = M = 51
dx, dy = 1.0 / (N - 1.5), 1.0 / (M - 1.5)
dt = 0.005
Dx, Dy = dt / dx ** 2, dt / dy ** 2
D = dx ** 2 / dt
D_f = dt / (2 * dx ** 2)
Tmax = 4
Nt = int(Tmax / dt)
# x = np.longfloat([(i - 1.5)*dx for i in range(1, N+1)])
# y = np.longfloat([(i - 1.5)*dx for i in range(1, M+1)])
x = np.linspace(0, 1, N, dtype=np.longfloat)
y = np.linspace(0, 1, M, dtype=np.longfloat)
X, Y = np.meshgrid(x, y, indexing='ij')
TITLES = {
    '5pt': '5-точечная схема',
    '5pt+t': '5-точечная схема Кранка-Николсона',
    '9pt': '9-точечная схема',
    '9pt+t': '9-точечная схема Кранка-Николсона'
}
U_numerical = {'5pt': [], '5pt+t': [], '9pt': [], '9pt+t': []}
U_exact = []
U_diff = {'5pt': [], '5pt+t': [], '9pt': [], '9pt+t': []}
U_heb = {'5pt': [], '5pt+t': [], '9pt': [], '9pt+t': []}


def run_process(keyid):
    for t in range(Nt):
        if t % (Nt / (5-1) / 10) == 0:
            print(f'Схема: {TITLES[keyid]}, t = {t * dt}')
        # Применение условий Дирихле
        U[keyid][:, -1] = 0
        U[keyid][-1, :] = 0
        if t == 250:
            print()
        if keyid in ['5pt', '9pt']:
            U_1D[keyid] = D * U[keyid].ravel()
            U_1D[keyid], exitcode = bicgstab(A[keyid].tocsr(), U_1D[keyid], tol=1e-9)
            if exitcode != 0:
                print(exitcode)
        if keyid in ['5pt+t', '9pt+t']:
            F[keyid] = B[keyid].dot(U[keyid].ravel())
            U_1D[keyid], exitcode = bicgstab(A[keyid].tocsr(), F[keyid], tol=1e-9)
        U[keyid] = U_1D[keyid].reshape(N, M)
        U_numerical[keyid].append(U[keyid])
        U_diff[keyid].append(
            np.abs(A[keyid].dot((U_exact[t] - U_numerical[keyid][t]).ravel()).reshape(N, M)))
        U_numerical[keyid][t] = U_numerical[keyid][t]
        # U_diff[keyid][t] = LU[keyid].solve(U_diff[keyid][t].flatten()).reshape(N, M)
        U_heb[keyid].append(np.mean(np.abs(U_exact[t] - U_numerical[keyid][t])) / N / M)
    return 0


def create_solution_graphs(keyid, tidx):
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
                im = ax.imshow(U_numerical[keyid][tidx], origin='lower', cmap='plasma', extent=[0, 1, 0, 1])
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
                im = ax.imshow(U_diff[keyid][tidx], origin='lower', cmap='plasma', extent=[0, 1, 0, 1])
                ax.set_title('Разность решений')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.cax.colorbar(im)
                ax.cax.toggle_label(True)
    fig.suptitle(f'Схема: {TITLES[keyid]}, t = {tidx * dt:0f}')
    plt.tight_layout()
    plt.savefig(f'unsteady_{keyid}_{tidx}')
    plt.close(fig)


def create_heb_graphs(keyid):
    plt.figure()
    plt.plot(np.arange(1, Nt) * dt, U_heb[keyid][1:])
    plt.title(f'Схема: {TITLES[keyid]}')
    plt.xlabel('t')
    plt.gca().set_yscale('log')
    plt.ylabel('HEB')
    plt.savefig(f'unsteady_{keyid}_heb')
    plt.close()


def analytical_solution(t):
    global X, Y
    result = np.exp(-np.pi ** 2 / 2 * t) * np.sin(np.pi / 2 * (1 - X)) * np.sin(np.pi / 2 * (1 - Y))
    return np.longfloat(result)


if __name__ == '__main__':
    # Начальное условие
    U = {
        '5pt': np.longfloat(np.sin(np.pi / 2 * (1 - X)) * np.sin(np.pi / 2 * (1 - Y))),
        '5pt+t': np.longfloat(np.sin(np.pi / 2 * (1 - X)) * np.sin(np.pi / 2 * (1 - Y))),
        '9pt': np.longfloat(np.sin(np.pi / 2 * (1 - X)) * np.sin(np.pi / 2 * (1 - Y))),
        '9pt+t': np.longfloat(np.sin(np.pi / 2 * (1 - X)) * np.sin(np.pi / 2 * (1 - Y)))
    }
    # Конструкция матрицы A для системы A*U_next = U_current
    A = {
        '5pt': lil_matrix((N * M, N * M), dtype=np.longfloat),
        '5pt+t': lil_matrix((N * M, N * M), dtype=np.longfloat),
        '9pt': lil_matrix((N * M, N * M), dtype=np.longfloat),
        '9pt+t': lil_matrix((N * M, N * M), dtype=np.longfloat)
    }
    B = {
        '5pt+t': lil_matrix((N * M, N * M), dtype=np.longfloat),
        '9pt+t': lil_matrix((N * M, N * M), dtype=np.longfloat)
    }
    F = {
        '5pt+t': np.zeros(N * M, dtype=np.longfloat),
        '9pt+t': np.zeros(N * M, dtype=np.longfloat)
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

            A['5pt'][idx, idx] = D + 4

            if i == N - 1 or j == M - 1:  # Правая и верхняя границы (условие Дирихле)
                A['5pt'][idx, idx] = 1
                continue

            if i == 0 and j == 0:  # Угловая точка (0, 0)
                A['5pt'][idx, idx] = D + 2
                A['5pt'][idx, idx + 1] = A['5pt'][idx, idx + N] = -1
                continue

            if i == 0:  # Левая граница
                A['5pt'][idx, idx] = D + 3
                A['5pt'][idx, idx + 1] = -1
                A['5pt'][idx, idx - N] = A['5pt'][idx, idx + N] = -1


            elif j == 0:  # Нижняя граница
                A['5pt'][idx, idx] = D + 3
                A['5pt'][idx, idx + N] = -1
                A['5pt'][idx, idx - 1] = A['5pt'][idx, idx + 1] = -1

            else:  # Внутренние узлы
                A['5pt'][idx, idx - 1] = A['5pt'][idx, idx + 1] = A['5pt'][idx, idx - N] = A['5pt'][idx, idx + N] = -1

    for i in range(N):
        for j in range(M):
            idx = i + j * N

            A['5pt+t'][idx, idx] = D + 2
            B['5pt+t'][idx, idx] = D - 2

            if i == N - 1 or j == M - 1:  # Правая и верхняя границы (условие Дирихле)
                A['5pt+t'][idx, idx] = 1
                B['5pt+t'][idx, idx] = 1
                continue

            if i == 0 and j == 0:
                A['5pt+t'][idx, idx + 1] = A['5pt+t'][idx, idx + N] = - 1
                B['5pt+t'][idx, idx + 1] = B['5pt+t'][idx, idx + N] = 1
                continue

            # Левая граница (условие Неймана)
            if i == 0:
                A['5pt+t'][idx, idx + 1] = - 1
                B['5pt+t'][idx, idx + 1] = 1
                A['5pt+t'][idx, idx - N] = A['5pt+t'][idx, idx + N] = -0.5
                B['5pt+t'][idx, idx - N] = B['5pt+t'][idx, idx + N] = 0.5

            # Нижняя граница (условие Неймана)
            elif j == 0:
                A['5pt+t'][idx, idx + N] = - 1
                B['5pt+t'][idx, idx + N] = 1
                A['5pt+t'][idx, idx - 1] = A['5pt+t'][idx, idx + 1] = -0.5
                B['5pt+t'][idx, idx - 1] = B['5pt+t'][idx, idx + 1] = 0.5

            # Верхняя и правая границы (условие Дирихле)
            else:
                A['5pt+t'][idx, idx + 1] = A['5pt+t'][idx, idx - 1] = A['5pt+t'][idx, idx + N] = A['5pt+t'][
                    idx, idx - N] = - 1 / 2
                B['5pt+t'][idx, idx + 1] = B['5pt+t'][idx, idx - 1] = B['5pt+t'][idx, idx + N] = B['5pt+t'][
                    idx, idx - N] = 1 / 2

    # Dxx = np.longfloat(dt / 6 / dx ** 2)
    for i in range(N):
        for j in range(M):
            idx = i + j * N

            A['9pt'][idx, idx] = D + 20 / 6

            if i == 0 and j == 0:
                A['9pt'][idx, idx + 1] = -8 / 6
                A['9pt'][idx, idx + N] = -8 / 6
                A['9pt'][idx, idx + N + 1] = - 4 / 6
                continue

            if i == N - 1 or j == M - 1:  # Правая и верхняя границы (условие Дирихле)
                A['9pt'][idx, idx] = 1

            # Левая граница (условие Неймана)
            elif i == 0:
                A['9pt'][idx, idx + 1] = -8 / 6
                A['9pt'][idx, idx - N] = A['9pt'][idx, idx + N] = -4 / 6
                A['9pt'][idx, idx - N + 1] = A['9pt'][idx, idx + N + 1] = - 2 / 6

            # Нижняя граница (условие Неймана)
            elif j == 0:
                A['9pt'][idx, idx + N] = -8 / 6
                A['9pt'][idx, idx - 1] = A['9pt'][idx, idx + 1] = - 4 / 6
                A['9pt'][idx, idx + N - 1] = A['9pt'][idx, idx + N + 1] = - 2 / 6

            # Верхняя и правая границы (условие Дирихле)
            else:
                # ближайшие соседи
                A['9pt'][idx, idx + 1] = A['9pt'][idx, idx - 1] = A['9pt'][idx, idx + N] = A['9pt'][
                    idx, idx - N] = - 4 / 6

                # диагональные элементы
                A['9pt'][idx, idx + N + 1] = A['9pt'][idx, idx - N - 1] = A['9pt'][idx, idx + N - 1] = A['9pt'][
                    idx, idx - N + 1] = - 1 / 6

    for j in range(M):
        for i in range(N):
            idx = i + j * N

            A['9pt+t'][idx, idx] = D + 10 / 6
            B['9pt+t'][idx, idx] = D - 10 / 6

            if i == 0 and j == 0:
                A['9pt+t'][idx, idx + 1] = A['9pt+t'][idx, idx + N] = -4 / 6
                B['9pt+t'][idx, idx + 1] = B['9pt+t'][idx, idx + N] = 4 / 6
                A['9pt+t'][idx, idx + N + 1] = - 2 / 6
                B['9pt+t'][idx, idx + N + 1] = 2 / 6

                continue

            if i == N - 1 or j == M - 1:  # Правая и верхняя границы (условие Дирихле)
                A['9pt+t'][idx, :] = 0
                B['9pt+t'][idx, :] = 0
                A['9pt+t'][idx, idx] = 1
                B['9pt+t'][idx, idx] = 1

            # Левая граница (условие Неймана)
            elif i == 0:
                A['9pt+t'][idx, idx + 1] = -4 / 6
                B['9pt+t'][idx, idx + 1] = 4 / 6
                A['9pt+t'][idx, idx - N] = A['9pt+t'][idx, idx + N] = -2 / 6
                B['9pt+t'][idx, idx - N] = B['9pt+t'][idx, idx + N] = 2 / 6
                A['9pt+t'][idx, idx - N + 1] = A['9pt+t'][idx, idx + N + 1] = - 1 / 6
                B['9pt+t'][idx, idx - N + 1] = B['9pt+t'][idx, idx + N + 1] = 1 / 6

            # Нижняя граница (условие Неймана)
            elif j == 0:
                A['9pt+t'][idx, idx + N] = -4 / 6
                B['9pt+t'][idx, idx + N] = 4 / 6
                A['9pt+t'][idx, idx - 1] = A['9pt+t'][idx, idx + 1] = -2 / 6
                B['9pt+t'][idx, idx - 1] = B['9pt+t'][idx, idx + 1] = 2 / 6
                A['9pt+t'][idx, idx + N - 1] = A['9pt+t'][idx, idx + N + 1] = - 1 / 6
                B['9pt+t'][idx, idx + N - 1] = B['9pt+t'][idx, idx + N + 1] = 1 / 6

            else:
                # ближайшие соседи
                A['9pt+t'][idx, idx + 1] = A['9pt+t'][idx, idx - 1] = A['9pt+t'][idx, idx + N] = A['9pt+t'][
                    idx, idx - N] = -2 / 6
                B['9pt+t'][idx, idx + 1] = B['9pt+t'][idx, idx - 1] = B['9pt+t'][idx, idx + N] = B['9pt+t'][
                    idx, idx - N] = 2 / 6

                # диагональные элементы
                A['9pt+t'][idx, idx + N + 1] = A['9pt+t'][idx, idx - N - 1] = A['9pt+t'][idx, idx + N - 1] = A['9pt+t'][
                    idx, idx - N + 1] = - 0.5 / 6
                B['9pt+t'][idx, idx + N + 1] = B['9pt+t'][idx, idx - N - 1] = B['9pt+t'][idx, idx + N - 1] = B['9pt+t'][
                    idx, idx - N + 1] = 0.5 / 6

    for key in ['5pt+t', '9pt+t']:
        B[key] = B[key].tocsc()
    # for key in ['5pt', '5pt+t', '9pt', '9pt+t']:
    #     LU[key] = splu(A[key].tocsc())
    key = ['5pt', '5pt+t', '9pt', '9pt+t']
    # key = ['5pt', '9pt']
    for t in range(Nt):
        U_exact.append(analytical_solution(t * dt))
    for item in key:
        U_numerical[item].append(U[item])
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = list(executor.map(run_process, key))
    for item in key:
        for i in range(1, Nt, int(Nt / (5 - 1))):
            # i -= 1
            create_solution_graphs(item, i)
        create_heb_graphs(item)
    plt.figure()
    for item in key:
        plt.plot(np.arange(1, Nt) * dt, U_heb[item][1:], label=item)
    plt.title('HEB')
    plt.legend()
    plt.gca().set_yscale('log')
    plt.xlabel('t')
    plt.savefig('unsteady_heb')
