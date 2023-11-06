import numpy as np
from matplotlib import gridspec
from scipy.sparse import lil_matrix, diags, identity
import concurrent.futures
from scipy.sparse.linalg import spsolve, splu
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm

# Параметры сетки и времени
N = M = 52
dx, dy = np.longfloat(1.0 / (N - 1.5)), np.longfloat(1.0 / (M - 1.5))
dt = np.longfloat(0.05)
Dx, Dy = np.longfloat(dt / dx ** 2), np.longfloat(dt / dy ** 2)
D = np.longfloat(dt / dx ** 2)
D_f = np.longfloat(dt / (2 * dx ** 2))
Tmax = 4
Nt = int(Tmax / dt)
x = np.longfloat([(i - 1.5)*dx for i in range(1, N+1)])
y = np.longfloat([(i - 1.5)*dx for i in range(1, M+1)])
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
        if t % 1000 == 0:
            print(f'Схема: {TITLES[keyid]}, t = {t * dt}')
        # Применение условий Дирихле
        U[keyid][:, -1] = 0
        U[keyid][-1, :] = 0
        if keyid in ['5pt', '9pt']:
            U_1D[keyid] = U[keyid].ravel()
            U_1D[keyid] = LU[keyid].solve(U_1D[keyid])
        if keyid in ['5pt+t']:
            F[keyid] = compute_b_5pt(U[keyid])
            U_1D[keyid] = LU[keyid].solve(F[keyid])
        if keyid in ['9pt+t']:
            F[keyid] = compute_b_9pt(U[keyid])
            U_1D[keyid] = LU[keyid].solve(F[keyid])
        U[keyid] = U_1D[keyid].reshape(N, M)
        U_numerical[keyid].append(U[keyid])
        U_diff[keyid].append(np.abs(U_exact[t] - U_numerical[keyid][t]))
        # U_diff[keyid][t] = LU[keyid].solve(U_diff[keyid][t].flatten()).reshape(N, M)
        U_heb[keyid].append(np.max(np.abs(U_exact[t] - U_numerical[keyid][t])) / N / M)
    return 0

def compute_b_5pt(u):
    global N, M

    b = np.zeros_like(u, dtype=np.longfloat)

    # Внутренние узлы
    b[1:-1, 1:-1] = u[1:-1, 1:-1] + \
                    0.5 * Dx * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1] + \
                                u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2])

    # Левая граница
    b[0, 1:-1] = u[0, 1:-1] + \
                 0.5 * Dx * (2 * u[1, 1:-1] - 2 * u[0, 1:-1] + \
                             u[0, 2:] - 2 * u[0, 1:-1] + u[0, :-2])

    # Нижняя граница
    b[1:-1, 0] = u[1:-1, 0] + \
                 0.5 * Dx * (u[2:, 0] - 2 * u[1:-1, 0] + u[:-2, 0] + \
                             2 * u[1:-1, 1] - 2 * u[1:-1, 0])

    # Точка (0, 0)
    b[0, 0] = u[0, 0] + Dx * (u[1, 0] - u[0, 0] + u[0, 1] - u[0, 0])

    b[-1, :] = 0
    b[:, -1] = 0

    return b.flatten()


def compute_b_9pt(u):
    global N, M

    b = np.zeros_like(u, dtype=np.longfloat)

    # Внутренние узлы
    b[1:-1, 1:-1] = u[1:-1, 1:-1] + \
                    0.5 * Dxx * (4 * (u[2:, 1:-1] + u[:-2, 1:-1] + \
                                u[1:-1, 2:] + u[1:-1, :-2]) + u[2:, 2:] + u[2:, :-2] + \
                                 u[:-2, 2:] + u[:-2, :-2] - 20 * u[1:-1, 1:-1])

    # Левая граница
    b[0, 1:-1] = u[0, 1:-1] + \
                 0.5 * Dxx * (8 * u[1, 1:-1] - 20 * u[0, 1:-1] + \
                             4 * u[0, 2:] + 4 * u[0, :-2] + 2 * u[1, 2:] + 2 * u[1, :-2])

    # Нижняя граница
    b[1:-1, 0] = u[1:-1, 0] + \
                 0.5 * Dxx * (4 * u[2:, 0] - 20 * u[1:-1, 0] + 4 * u[:-2, 0] + \
                             8 * u[1:-1, 1] + 2 * u[2:, 1] + 2 * u[:-2, 1])

    # Точка (0, 0)
    b[0, 0] = u[0, 0] + 0.5 * Dxx * (8 * u[1, 0] - 20 * u[0, 0] + 8 * u[0, 1] + 4 * u[1, 1])

    b[-1, :] = 0
    b[:, -1] = 0

    return b.flatten()


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
    fig.suptitle(f'Схема: {TITLES[keyid]}, t = {tidx * dt:3f}')
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
    return result


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
        '5pt': lil_matrix(((N-1) * (M-1), (N-1) * (M-1)), dtype=np.longfloat),
        '5pt+t': lil_matrix((N * M, N * M), dtype=np.longfloat),
        '9pt': lil_matrix((N * M, N * M), dtype=np.longfloat),
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

    for i in range(N-1):
        for j in range(M-1):
            idx = i + j * (N-1)

            A['5pt'][idx, idx] = 1 + 4 * Dx

            if i == N - 2 or j == M - 2:  # Правая и верхняя границы (условие Дирихле)
                A['5pt'][idx, idx] = 1
                continue

            if i == 0 and j == 0:  # Угловая точка (0, 0)
                A['5pt'][idx, idx + 1] = A['5pt'][idx, idx + N-1] = -2 * Dx
                continue

            if i == 0:  # Левая граница
                A['5pt'][idx, idx + 1] = -2 * Dx
                A['5pt'][idx, idx - N+1] = A['5pt'][idx, idx + N-1] = -Dx


            elif j == 0:  # Нижняя граница
                A['5pt'][idx, idx + N-1] = -2 * Dy
                A['5pt'][idx, idx - 1] = A['5pt'][idx, idx + 1] = -Dx

            else:  # Внутренние узлы
                A['5pt'][idx, idx - 1] = A['5pt'][idx, idx + 1] = A['5pt'][idx, idx - N+1] = A['5pt'][idx, idx + N-1] = -Dx

    for i in range(N):
        for j in range(N):
            idx = i + j * N

            A['5pt+t'][idx, idx] = 1 + 2 * Dx

            if i == N - 1 or j == M - 1:  # Правая и верхняя границы (условие Дирихле)
                A['5pt+t'][idx, idx] = 1
                continue

            if i == 0 and j == 0:
                A['5pt+t'][idx, idx + 1] = A['5pt+t'][idx, idx + N] = - 1 * Dx
                continue

            # Левая граница (условие Неймана)
            if i == 0:
                A['5pt+t'][idx, idx + 1] = - 1 * Dx
                A['5pt+t'][idx, idx - N] = A['5pt+t'][idx, idx + N] = -0.5 * Dx

            # Нижняя граница (условие Неймана)
            elif j == 0:
                A['5pt+t'][idx, idx + N] = - 1 * Dx
                A['5pt+t'][idx, idx - 1] = A['5pt+t'][idx, idx + 1] = -0.5 * Dx

            # Верхняя и правая границы (условие Дирихле)
            else:
                A['5pt+t'][idx, idx + 1] = A['5pt+t'][idx, idx - 1] = A['5pt+t'][idx, idx + N] = A['5pt+t'][
                    idx, idx - N] = - Dx / 2

    Dxx = np.longfloat(dt / 6 / dx ** 2)
    for i in range(N):
        for j in range(M):
            idx = i + j * N

            A['9pt'][idx, idx] = 1 + 20 * Dxx

            if i == 0 and j == 0:
                A['9pt'][idx, idx + 1] = -8 * Dxx
                A['9pt'][idx, idx + N] = -8 * Dxx
                A['9pt'][idx, idx + N + 1] = - 4 * Dxx
                continue

            if i == N - 1 or j == M - 1:  # Правая и верхняя границы (условие Дирихле)
                A['9pt'][idx, idx] = 1

            # Левая граница (условие Неймана)
            elif i == 0:
                A['9pt'][idx, idx + 1] = -8 * Dxx
                A['9pt'][idx, idx - N] = A['9pt'][idx, idx + N] = -4 * Dxx
                A['9pt'][idx, idx - N + 1] = A['9pt'][idx, idx + N + 1] = - 2 * Dxx

            # Нижняя граница (условие Неймана)
            elif j == 0:
                A['9pt'][idx, idx + N] = -8 * Dxx
                A['9pt'][idx, idx - 1] = A['9pt'][idx, idx + 1] = - 4 * Dxx
                A['9pt'][idx, idx + N - 1] = A['9pt'][idx, idx + N + 1] = - 2 * Dxx

            # Верхняя и правая границы (условие Дирихле)
            else:
                # ближайшие соседи
                A['9pt'][idx, idx + 1] = A['9pt'][idx, idx - 1] = A['9pt'][idx, idx + N] = A['9pt'][
                    idx, idx - N] = - 4 * Dxx

                # диагональные элементы
                A['9pt'][idx, idx + N + 1] = A['9pt'][idx, idx - N - 1] = A['9pt'][idx, idx + N - 1] = A['9pt'][
                    idx, idx - N + 1] = - 1 * Dxx

    for j in range(M):
        for i in range(N):
            idx = i + j * N

            A['9pt+t'][idx, idx] = 1 + 10 * Dxx

            if i == 0 and j == 0:
                A['9pt+t'][idx, idx + 1] = A['9pt+t'][idx, idx + N] = -4 * Dxx
                A['9pt+t'][idx, idx + N + 1] = - 2 * Dxx

                continue

            if i == N - 1 or j == M - 1:  # Правая и верхняя границы (условие Дирихле)
                A['9pt+t'][idx, :] = 0
                A['9pt+t'][idx, idx] = 1

            # Левая граница (условие Неймана)
            elif i == 0:
                A['9pt+t'][idx, idx + 1] = -4 * Dxx
                A['9pt+t'][idx, idx - N] = A['9pt+t'][idx, idx + N] = -2 * Dxx
                A['9pt+t'][idx, idx - N + 1] = A['9pt+t'][idx, idx + N + 1] = - 1 * Dxx

            # Нижняя граница (условие Неймана)
            elif j == 0:
                A['9pt+t'][idx, idx + N] = -4 * Dxx
                A['9pt+t'][idx, idx - 1] = A['9pt+t'][idx, idx + 1] = -2 * Dxx
                A['9pt+t'][idx, idx + N - 1] = A['9pt+t'][idx, idx + N + 1] = - 1 * Dxx


            else:
                # ближайшие соседи
                A['9pt+t'][idx, idx + 1] = A['9pt+t'][idx, idx - 1] = A['9pt+t'][idx, idx + N] = A['9pt+t'][
                    idx, idx - N] = -2 * Dxx

                # диагональные элементы
                A['9pt+t'][idx, idx + N + 1] = A['9pt+t'][idx, idx - N - 1] = A['9pt+t'][idx, idx + N - 1] = A['9pt+t'][
                    idx, idx - N + 1] = - 0.5 * Dxx

    F['5pt+t'] = compute_b_5pt(U['5pt+t'])
    F['9pt+t'] = compute_b_9pt(U['9pt+t'])
    for key in ['5pt', '5pt+t', '9pt', '9pt+t']:
        LU[key] = splu(A[key].tocsc())

    # key = ['5pt', '5pt+t', '9pt', '9pt+t']
    key = ['5pt']
    # U_exact.append(U['5pt'])
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

