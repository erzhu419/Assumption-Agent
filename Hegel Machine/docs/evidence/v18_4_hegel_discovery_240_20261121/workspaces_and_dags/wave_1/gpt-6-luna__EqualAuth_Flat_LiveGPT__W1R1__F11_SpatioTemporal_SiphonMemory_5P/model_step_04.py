import math
import numpy as np
from scipy.optimize import least_squares

_FIT_PARAMS = None

def _simulate(row, params):
    a_slow, b_couple, c_drive, tau_fast = params
    h = float(row['H'])
    n = max(1, int(math.ceil(h / 0.004)))
    dt = h / n
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u = float(row['u_now'])
    def deriv(xv, zv):
        return -a_slow * xv + b_couple * zv + c_drive * u, (0.52 * xv - zv) / tau_fast
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5 * dt * k1x, z + 0.5 * dt * k1z)
        k3x, k3z = deriv(x + 0.5 * dt * k2x, z + 0.5 * dt * k2z)
        k4x, k4z = deriv(x + dt * k3x, z + dt * k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6.0
    return x

def fit(train_rows):
    global _FIT_PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_simulate(r, p) for r in rows]) - y
    best = None
    starts = ([0.5, 0.2, 0.2, 0.5], [1.0, 0.5, 0.5, 1.0], [0.2, 0.8, 0.1, 0.2], [1.5, 0.1, 0.8, 2.0])
    for start in starts:
        result = least_squares(residual, start, bounds=([0.0, -5.0, -5.0, 0.02], [10.0, 5.0, 5.0, 10.0]), max_nfev=3000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    _FIT_PARAMS = best.x
    return _FIT_PARAMS

def predict(row):
    if _FIT_PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_simulate(row, _FIT_PARAMS))