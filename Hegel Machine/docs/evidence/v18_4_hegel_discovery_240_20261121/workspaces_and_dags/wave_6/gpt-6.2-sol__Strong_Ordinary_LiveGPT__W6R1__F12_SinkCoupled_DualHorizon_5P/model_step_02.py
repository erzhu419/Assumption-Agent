import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, h, y = np.array([(r['u_now'], r['s_grad'], r['H'], r['y']) for r in train_rows], dtype=float).T
    sink = s / (1.0 + s)
    sat = u / (1.2 + u)

    def residual(p):
        vmax, rho, tau_fast, amp_adapt, tau_slow = p
        fast = -np.expm1(-h / tau_fast)
        slow = -np.expm1(-h / tau_slow)
        return vmax * (1.0 - rho * sink) * sat * fast - amp_adapt * u * slow - y

    result = least_squares(residual, [7.0, 0.7, 0.3, 2.0, 10.0], bounds=([0.0, 0.0, 1e-6, 0.0, 1e-6], [np.inf, 1.0, np.inf, np.inf, np.inf]), x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x

def predict(row) -> float:
    vmax, rho, tau_fast, amp_adapt, tau_slow = _params
    u, s, h = row['u_now'], row['s_grad'], row['H']
    fast = -math.expm1(-h / tau_fast)
    slow = -math.expm1(-h / tau_slow)
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) * fast - amp_adapt * u * slow)
