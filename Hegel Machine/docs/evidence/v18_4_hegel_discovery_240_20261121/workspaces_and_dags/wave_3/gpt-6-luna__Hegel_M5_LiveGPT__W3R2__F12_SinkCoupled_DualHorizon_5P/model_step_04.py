import math
import numpy as np
from scipy.optimize import least_squares

def _simulate(row, p):
    a, b, vmax, rho, tau = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = max(0.0, float(row['H']))
    x = float(row['x_now'])
    z = float(row.get('mem_conv') or 0.0)
    J = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    n = max(1, int(math.ceil(H / 0.02)))
    dt = H / n
    def f(xv, zv):
        return -a * xv + b * zv + J, (0.52 * xv - zv) / tau
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = f(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = f(x + dt*k3x, z + dt*k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6.0
    return x

def fit(train_rows):
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_simulate(r, p) for r in train_rows]) - y
    starts = [[0.2, 0.1, 5.0, 0.7, 1.0], [0.5, 0.2, 8.0, 0.5, 3.0], [0.05, 0.01, 4.0, 0.7, 0.3]]
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=([1e-5, -5.0, 0.01, 0.0, 0.01], [5.0, 5.0, 30.0, 0.99, 50.0]), max_nfev=5000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    return best.x

def predict(row):
    return float(_simulate(row, _PARAMS))