import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _simulate(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    H = float(row['H'])
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u_n = float(row['u_now']) / 2.0
    s = float(row['s_grad'])
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    n = max(20, int(math.ceil(200 * H)))
    dt = H / n
    def f(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + drive, (0.52 * xx - zz) / tau_fast])
    state = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return float(state[0])

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_simulate(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([0.4, 0.3, 6.0, 0.7, 1.0]), bounds=([0.001, -2.0, 0.01, 0.0, 0.05], [4.0, 2.0, 20.0, 0.99, 10.0]), max_nfev=5000)
    _PARAMS = result.x.copy()
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return _simulate(row, _PARAMS)
