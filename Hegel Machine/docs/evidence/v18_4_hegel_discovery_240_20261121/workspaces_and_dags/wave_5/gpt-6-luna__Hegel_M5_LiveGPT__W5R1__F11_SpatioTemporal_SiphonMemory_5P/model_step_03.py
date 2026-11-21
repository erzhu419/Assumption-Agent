import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _simulate(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    u_n = float(row['u_now']) / 2.0
    s = float(row['s_grad'])
    H = float(row['H'])
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    j = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    n = max(1, int(math.ceil(H / 0.005)))
    dt = H / n
    def f(xv, zv):
        return (-a_slow * xv + b_couple * zv + j, (0.52 * xv - zv) / tau_fast)
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = f(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = f(x + dt*k3x, z + dt*k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6.0
    return x

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_simulate(r, p) for r in train_rows]) - y
    x0 = np.array([0.5, 0.2, 6.0, 0.7, 0.8])
    lo = np.array([0.001, -3.0, 0.01, 0.001, 0.01])
    hi = np.array([5.0, 3.0, 30.0, 0.999, 10.0])
    result = least_squares(residual, x0, bounds=(lo, hi), max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise ValueError('Call fit(train_rows) before predict(row).')
    return float(_simulate(row, _PARAMS))