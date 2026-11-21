import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, g, h, y = np.asarray([[r['u_now'], r['s_grad'], r['H'], r['y']] for r in train_rows], dtype=float).T
    saturation = u / (1.2 + u)
    sink = g / (1.0 + g)
    def residual(p):
        vmax, rho, tau_fast, amp, tau_slow = p
        fast = -np.expm1(-h / tau_fast)
        slow = -np.expm1(-h / tau_slow)
        return vmax * (1.0 - rho * sink) * saturation * fast - amp * u * slow - y
    _params = least_squares(residual, [5.0, 0.5, 0.5, 1.0, 10.0], bounds=([0.0, 0.0, 1e-6, 0.0, 1e-6], [np.inf, 1.0, np.inf, np.inf, np.inf]), x_scale='jac').x

def predict(row) -> float:
    vmax, rho, tau_fast, amp, tau_slow = _params
    u, g, h = row['u_now'], row['s_grad'], row['H']
    fast = -math.expm1(-h / tau_fast)
    slow = -math.expm1(-h / tau_slow)
    return float(vmax * (1.0 - rho * g / (1.0 + g)) * u / (1.2 + u) * fast - amp * u * slow)
