import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_TAU_SCALE = 1.0

def _predict_with(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    u_n = float(row['u_now']) / _TAU_SCALE
    s = float(row['s_grad'])
    H = float(row['H'])
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    # RK4 for dx/dt=-a_slow*x+b_couple*z+drive; dz/dt=(0.52*x-z)/tau_fast.
    n = max(40, int(math.ceil(240.0 * H)))
    dt = H / n
    def f(xx, zz):
        return -a_slow * xx + b_couple * zz + drive, (0.52 * xx - zz) / tau_fast
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = f(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = f(x + dt*k3x, z + dt*k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6.0
    return float(x)

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_with(r, p) for r in train_rows]) - y
    p0 = np.array([0.7, 0.5, 6.0, 0.7, 0.8], dtype=float)
    lower = np.array([0.01, -2.0, 0.05, 0.01, 0.03], dtype=float)
    upper = np.array([5.0, 3.0, 20.0, 0.99, 5.0], dtype=float)
    result = least_squares(residual, p0, bounds=(lower, upper), max_nfev=3000)
    _PARAMS = result.x
    return _PARAMS.copy()

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return _predict_with(row, _PARAMS)
