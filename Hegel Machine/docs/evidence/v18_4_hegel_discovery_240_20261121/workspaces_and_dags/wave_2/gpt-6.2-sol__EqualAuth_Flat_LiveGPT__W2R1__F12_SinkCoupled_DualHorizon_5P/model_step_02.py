import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, h, y = np.asarray([[r['u_now'], r['s_grad'], r['H'], r['y']] for r in train_rows], dtype=float).T
    drive = u / (1.2 + u)
    sink = s / (1.0 + s)

    def residual(p):
        vmax, rho, tau_fast, amp, gap = p
        tau_slow = tau_fast + gap
        fast = -np.expm1(-h / tau_fast)
        slow = -np.expm1(-h / tau_slow)
        return vmax * (1.0 - rho * sink) * drive * fast - amp * u * slow - y

    _params = least_squares(
        residual, [7.0, 0.6, 0.4, 1.5, 12.0],
        bounds=([0.0, 0.0, 1e-6, 0.0, 1e-6], [np.inf, 1.0, np.inf, np.inf, np.inf]),
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
    ).x

def predict(row):
    vmax, rho, tau_fast, amp, gap = _params
    u, s, h = (row[k] for k in ('u_now', 's_grad', 'H'))
    fast = -math.expm1(-h / tau_fast)
    slow = -math.expm1(-h / (tau_fast + gap))
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * fast - amp * u * slow)
