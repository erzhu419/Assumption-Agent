import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _simulate(row, p):
    a, b, vmax, rho, tau = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    x = float(row['x_now'])
    z = float(row['mem_conv'] or 0.0)
    H = float(row['H'])
    J = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    n = max(1, int(math.ceil(H / 0.08)))
    dt = H / n
    def f(xv, zv):
        return -a * xv + b * zv + J, (0.52 * xv - zv) / tau
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
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_simulate(r, p) for r in train_rows]) - ys
    best = None
    for start in ([0.2, 0.1, 3.0, 0.5, 1.0], [0.05, 0.02, 5.0, 0.7, 2.0]):
        res = least_squares(residual, np.array(start), bounds=([0.001, -2.0, 0.01, -1.0, 0.03], [3.0, 2.0, 20.0, 1.5, 20.0]), max_nfev=300)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    _PARAMS = best.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_simulate(row, _PARAMS))