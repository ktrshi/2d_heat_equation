import scipy
import numpy as np

N = M = 52
dx, dy = np.longfloat(1.0 / (N - 1.5)), np.longfloat(1.0 / (M - 1.5))
dt = np.longfloat(0.001)
Dx, Dy = np.longfloat(dt / dx ** 2), np.longfloat(dt / dy ** 2)
D = np.longfloat(dx ** 2 / dt)

A0 = scipy.sparse.lil_matrix((N, M))
A1 = scipy.sparse.lil_matrix((N, M))
A2 = scipy.sparse.lil_matrix((N, M))
A3 = scipy.sparse.lil_matrix((N, M))
A4 = scipy.sparse.lil_matrix((N, M))
A5 = scipy.sparse.lil_matrix((N, M))
A6 = scipy.sparse.lil_matrix((N, M))
A7 = scipy.sparse.lil_matrix((N, M))
A8 = scipy.sparse.lil_matrix((N, M))
A0[:, :] = D + 4
A1[:, :] = -1
A3[:, :] = -1
A5[:, :] = -1
A7[:, :] = -1

A7[:, -1] = 0

A0[1:-1, 1] += A7[1:-1, 1]
A7[:-1, 1] = 0

A5[-1, 1:-1] = 0

A0[1, 1:-1] += A5[1, 1:-1]
A5[1, 1:-1] = 0

top_row = scipy.sparse.hstack([A8[0, 0], A7[0, 0], A6[0, 0]])
middle_row = scipy.sparse.hstack([A1[0, 0], A0[0, 0], A5[0, 0]])
bottom_row = scipy.sparse.hstack([A2[0, 0], A3[0, 0], A4[0, 0]])
A = scipy.sparse.vstack([top_row, middle_row, bottom_row])

print(0)
