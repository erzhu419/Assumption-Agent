import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _simulate(row, p):
    a_slow, b_couple, c_drive, tau_fast = p
    H = max(0.0, float(row['H']))
    x = float(row['x_now'])
    z = float(row['mem_conv']) if row.get('mem_conv') is not None else 0.52 * x
    u = float(row['u_now'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def deriv(xv, zv):
        return -a_slow * xv + b_couple * zv + c_drive * u, (0.52 * xv - zv) / tau_fast
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = deriv(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_simulate(r, p) for r in rows]) - y
    starts = [np.array([0.5, 0.2, 0.5, 0.5]), np.array([1.0, 0.5, 1.0, 1.0]), np.array([0.2, 0.05, 0.2, 0.2])]
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=([0.0, -5.0, -5.0, 0.01], [10.0, 5.0, 5.0, 20.0]), max_nfev=5000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    _PARAMS = best.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise ValueError('fit must be called before predict')
    return float(_simulate(row, _PARAMS))