import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _simulate(row, p):
    vmax, rho, a_slow, b_couple, tau_fast = p
    u = row['u_now'] / 2.0
    s = row['s_grad']
    j = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(xv, zv):
        return -a_slow * xv + b_couple * zv + j, (0.52 * xv - zv) / tau_fast
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = f(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = f(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_simulate(r, p) for r in train_rows]) - y
    starts = [[1.0, 0.5, 0.3, 0.2, 0.5], [2.0, 0.3, 0.6, 0.5, 1.0], [0.5, 0.7, 0.1, 0.05, 0.2]]
    fits = [least_squares(residual, q, bounds=([0.0, 0.0, 0.0, 0.0, 0.01], [20.0, 1.0, 10.0, 10.0, 10.0]), max_nfev=5000) for q in starts]
    _PARAMS = min(fits, key=lambda q: np.sum(q.fun*q.fun)).x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit must be called first')
    return float(_simulate(row, _PARAMS))