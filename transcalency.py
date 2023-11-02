import numpy as np
import math


def initial_conditions(i, j):
    return np.sin(np.pi / 2 * (1 - (i - 1.5) * hx)) * np.sin(np.pi / 2 * (1 - (j - 1.5) * hx))


def solve(A, AL, HEB, DP, FIL, eps):
    zmaxheb, zmaxhe = 0, 0
    pp = np.zeros((N, M))
    pp_alt = np.zeros((N, M))
    r = np.zeros((N, M))
    it = np.arange(1, N - 2)
    jt = np.arange(1, M - 2)
    HEB[it[:, None], jt] = (
            A[0][it[:, None], jt] * DP[it[:, None], jt] +
            A[5][it[:, None] + 1, jt] * DP[it[:, None] + 1, jt] + A[5][it[:, None], jt] * DP[it[:, None] - 1, jt] +
            A[6][it[:, None] + 1, jt + 1] * DP[it[:, None] + 1, jt + 1] + A[6][it[:, None], jt] *
            DP[it[:, None] - 1, jt - 1] +
            A[7][it[:, None], jt + 1] * DP[it[:, None], jt + 1] + A[7][it[:, None], jt] * DP[it[:, None], jt - 1] +
            A[8][it[:, None] - 1, jt + 1] * DP[it[:, None] - 1, jt + 1] + A[8][it[:, None], jt] *
            DP[it[:, None] + 1, jt - 1] -
            HEB[it[:, None], jt]
    )
    zmaxhe = abs(np.max(HEB[1:N-1, 1:M-1]))
    zmaxheb = np.sum(np.square(HEB[1:N-1, 1:M-1]))
    zmaxheb = math.sqrt(zmaxheb / N / M)

    ltl(HEB, pp, FIL, AL)

    n_iter = 0
    while zmaxheb > eps:
        # print('n_iter = ', n_iter, 'zmaxheb = ', zmaxheb)
        n_iter += 1
        zpheb = 0
        zppr = 0
        r[it[:, None], jt] = (A[0][it[:, None], jt] * pp[it[:, None], jt] + A[5][it[:, None] + 1, jt] *
                              pp[it[:, None] + 1, jt] +
                              A[6][it[:, None] + 1, jt + 1] * pp[it[:, None] + 1, jt + 1] + A[7][it[:, None], jt + 1] *
                              pp[it[:, None], jt + 1] + A[8][it[:, None] - 1, jt + 1] * pp[it[:, None] - 1, jt + 1] +
                              A[5][it[:, None], jt] * pp[it[:, None] - 1, jt] + A[6][it[:, None], jt] *
                              pp[it[:, None] - 1, jt - 1] + A[7][it[:, None], jt] * pp[it[:, None], jt - 1] +
                              A[8][it[:, None], jt] * pp[it[:, None] + 1, jt - 1])
        zpheb = np.sum(HEB * pp)
        zppr = np.sum(r*pp)
        zak = zpheb / (zppr + 1e-20)
        zakp = zak * pp
        DP = DP - zakp
        HEB = HEB - zak * r
        zmaxhe = abs(np.max(HEB[1:N-1, 1:M-1]))
        zmaxheb = np.sum(np.square(HEB[1:N-1, 1:M-1]))
        zmaxheb = math.sqrt(zmaxheb / N / M)
        if zmaxheb < eps:
            break
        ltl(HEB, pp_alt, FIL, AL)
        zyyr = np.sum(pp_alt * r)
        zbk = zyyr / (zppr + 1e-20)
        pp = pp_alt - zbk * pp
    return DP


def ltl(F, X, fil, AL):
    it = np.arange(1, N - 2)
    jt = np.arange(1, M - 2)
    fil[it[:, None], jt] = (F[it[:, None], jt] - AL[5][it[:, None], jt] * fil[it[:, None] - 1, jt] -
                            AL[6][it[:, None], jt] * fil[it[:, None] - 1, jt - 1] -
                            AL[7][it[:, None], jt] * fil[it[:, None], jt - 1] - AL[8][it[:, None], jt] *
                            fil[it[:, None] + 1, jt - 1]) / AL[0][it[:, None], jt]
    it = np.arange(N - 2, 0, -1)
    jt = np.arange(M - 2, 0, -1)
    X[it[:, None], jt] = (fil[it[:, None], jt] - AL[5][it[:, None] + 1, jt] * X[it[:, None], jt] -
                          AL[6][it[:, None] + 1, jt + 1] * X[it[:, None] + 1, jt + 1] -
                          AL[7][it[:, None], jt + 1] * X[it[:, None], jt + 1] -
                          AL[8][it[:, None] - 1, jt + 1] * X[it[:, None] - 1, jt + 1]) / AL[0][it[:, None], jt]


def alt(A, AL, ALOK):
    it = np.arange(1, N - 1)
    jt = np.arange(1, M - 1)
    AL[6][it[:, None], jt] = A[6][it[:, None], jt] / (AL[0][it[:, None] - 1, jt - 1] + ALOK)
    AL[7][it[:, None], jt] = (A[7][it[:, None], jt] - AL[5][it[:, None], jt - 1] * AL[6][it[:, None], jt]) / (
            AL[0][it[:, None], jt - 1] + ALOK)
    AL[8][it[:, None], jt] = (A[8][it[:, None], jt] - AL[7][it[:, None], jt] * AL[5][it[:, None] + 1, jt - 1]) / (
            AL[0][it[:, None] + 1, jt - 1] + ALOK)
    AL[5][it[:, None], jt] = (A[5][it[:, None], jt] - AL[6][it[:, None], jt] * AL[7][it[:, None], jt] - AL[7][
        it[:, None], jt] * AL[8][it[:, None], jt]) / (AL[0][it[:, None], jt] + ALOK)
    AL[0] = np.sqrt(AL[0] - AL[5] ** 2 - AL[6] ** 2 - AL[7] ** 2 - AL[8] ** 2)
    zsq1 = 1 / np.sqrt(AL[0])
    AL[0] = AL[0] * zsq1
    AL[5][it[:, None] + 1, jt] = AL[5][it[:, None] + 1, jt] * zsq1[1:N - 1, 1:M - 1]
    AL[6][it[:, None] + 1, jt + 1] = AL[6][it[:, None] + 1, jt + 1] * zsq1[1:N - 1, 1:M - 1]
    AL[7][it[:, None], jt + 1] = AL[7][it[:, None], jt + 1] * zsq1[1:N - 1, 1:M - 1]
    AL[8][it[:, None] - 1, jt + 1] = AL[8][it[:, None] - 1, jt + 1] * zsq1[1:N - 1, 1:M - 1]


N, M = 202, 202
A = [np.zeros((N, M)) for _ in range(9)]
AL = [np.zeros((N, M)) for _ in range(9)]
F = np.zeros((N, M))
FIL = np.zeros((N, M))
eps = 1e-5
alok = 1e-1
tau = 1e-2
Nt = 100
time_max = 10
hx = 1 / (N - 0.5)
X = np.fromfunction(initial_conditions, (N, M), dtype=float)
Xn = X.copy()
A[0][:, :] = 4 + hx ** 2 / tau
AL[0][:, :] = 4
A[1][:, :] = -1
A[3][:, :] = -1
A[5][:, :] = -1
A[7][:, :] = -1
A[7][:, M - 1] = 0

A[0][:, 0] = A[0][:, 0] + A[7][:, 0]
A[7][:, 0] = 0

A[0][0, :] = A[0][0, :] + A[5][0, :]
A[5][0, :] = 0
A[5][N - 1, :] = 0
nt = 0
time = 0
alt(A, AL, alok)
for nt in range(Nt):
    print('nt = ', nt)
    F = hx ** 2 * Xn / tau
    Xn = solve(A, AL, F, Xn, FIL, eps)
    X = Xn
    X[:, 0] = X[:, 1]
    X[:, M - 1] = 0
    X[0, :] = X[1, :]
    X[N - 1, :] = 0
    Xn = X.copy()
print()
