import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    u_n = float(row['u_now']) / 0.005
    s = float(row['s_grad'])
    J = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def deriv(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + J,
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
        return np.array([_predict_with(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([0.5, 0.2, 1.0, 0.5, 0.01]),
                           bounds=([0.0, 0.0, 0.0, 0.0, 0.0001],
                                   [20.0, 20.0, 20.0, 0.999, 2.0]),
                           max_nfev=20000)
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return _predict_with(row, _params)