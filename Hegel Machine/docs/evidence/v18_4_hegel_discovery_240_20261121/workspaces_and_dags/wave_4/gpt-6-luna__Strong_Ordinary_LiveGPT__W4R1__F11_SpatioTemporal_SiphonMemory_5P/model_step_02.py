import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _simulate(row, p):
    vmax, rho, a_slow, b_couple, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    u_n = u  # characteristic tau_scale is 1.0
    J = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + J, (0.52 * xx - zz) / tau_fast])
    state = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return float(state[0])

def fit(train_rows):
    global _PARAMS
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_simulate(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([2.0, 0.5, 0.5, 0.2, 0.2]), bounds=(np.array([0.0, 0.0, 0.0, 0.0, 0.005]), np.array([20.0, 0.999, 10.0, 10.0, 5.0])), max_nfev=5000)
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return _simulate(row, _PARAMS)