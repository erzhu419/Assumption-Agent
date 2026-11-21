import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _simulate(row, p):
    a_slow, b_couple, c_drive, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(xv, zv):
        return -a_slow * xv + b_couple * zv + c_drive * u, (0.52 * xv - zv) / tau_fast
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
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_simulate(r, p) for r in rows]) - y
    best = None
    for start in ([0.2, 0.1, 1.0, 0.5], [0.5, 0.5, 2.0, 1.0], [1.0, 0.2, 1.0, 2.0]):
        res = least_squares(residual, np.asarray(start), bounds=([1e-5, -5.0, -5.0, 1e-3], [5.0, 5.0, 5.0, 20.0]), max_nfev=2000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    _PARAMS = best.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_simulate(row, _PARAMS))