import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(p, u, s, h, expm1):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    fast = -expm1(-h / tau_fast)
    slow = -expm1(-h / tau_slow)
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * fast - amp_adapt * u * slow

def fit(train_rows):
    global _params
    u, s, h, y = np.asarray([[r['u_now'], r['s_grad'], r['H'], r['y']] for r in train_rows], dtype=float).T
    result = least_squares(
        lambda p: _response(p, u, s, h, np.expm1) - y,
        x0=[5.0, 0.6, 0.3, 1.5, 10.0],
        bounds=([0.0, 0.0, 1e-6, 0.0, 1e-6], [np.inf, 1.0, np.inf, np.inf, np.inf]),
        x_scale='jac',
    )
    _params = result.x

def predict(row) -> float:
    return float(_response(_params, row['u_now'], row['s_grad'], row['H'], math.expm1))