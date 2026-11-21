import math
import numpy as np
from scipy.optimize import least_squares

def _simulate(row, p):
    a, b, beta, c = p
    u = float(row['u_now']) - beta * float(row['s_grad'])
    z1 = float(row['u_past'])
    z2 = float(row['x_now'])
    H = float(row['H'])
    n = max(20, int(math.ceil(100 * abs(H))))
    dt = H / n
    def f(q):
        x, y = q
        return np.array([a * (u - x), b * (x - y)])
    q = np.array([z1, z2], dtype=float)
    for _ in range(n):
        k1 = f(q)
        k2 = f(q + 0.5 * dt * k1)
        k3 = f(q + 0.5 * dt * k2)
        k4 = f(q + dt * k3)
        q = q + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return c * q[1]

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_simulate(r, p) for r in rows]) - y
    starts = [np.array([1.0, 1.0, 0.5, 1.0]), np.array([2.0, 0.5, -0.5, 2.0]), np.array([0.5, 2.0, 1.0, 1.0])]
    sols = [least_squares(residual, s, max_nfev=3000) for s in starts]
    best = min(sols, key=lambda z: float(np.dot(z.fun, z.fun)))
    return best.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_simulate(row, p))