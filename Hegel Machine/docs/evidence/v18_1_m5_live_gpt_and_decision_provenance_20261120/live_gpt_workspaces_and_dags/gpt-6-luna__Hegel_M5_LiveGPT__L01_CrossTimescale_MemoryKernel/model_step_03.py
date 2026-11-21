import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _simulate(row, p):
    a_slow, b_couple, c_drive, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def deriv(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + c_drive * u,
                         (0.52 * xx - zz) / tau_fast])
    state = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return float(state[0])

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_simulate(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([1.0, 0.5, 1.0, 0.2]),
                           bounds=([0.0, -10.0, -10.0, 0.001], [10.0, 10.0, 10.0, 10.0]),
                           max_nfev=5000)
    _params = result.x.copy()
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return _simulate(row, _params)
