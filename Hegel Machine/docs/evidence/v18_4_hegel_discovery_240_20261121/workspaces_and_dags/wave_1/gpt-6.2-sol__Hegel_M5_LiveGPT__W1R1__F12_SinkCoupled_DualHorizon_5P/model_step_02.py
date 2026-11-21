import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def fit(train_rows):
    global _params
    rows = list(train_rows)
    u, s, h, y = np.asarray([[r['u_now'], r['s_grad'], r['H'], r['y']] for r in rows], dtype=float).T
    drive = u / (1.2 + u)
    sink = s / (1.0 + s)

    def residual(p):
        vmax, rho, tau_fast, amp_adapt, gap = p
        fast = -np.expm1(-h / tau_fast)
        slow = -np.expm1(-h / (tau_fast + gap))
        return vmax * (1.0 - rho * sink) * drive * fast - amp_adapt * u * slow - y

    result = least_squares(residual, [6.5, 0.70, 0.25, 2.0, 15.0], bounds=([0.0, 0.0, 1e-6, 0.0, 1e-6], [np.inf, 1.0, np.inf, np.inf, np.inf]), x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x


def predict(row):
    vmax, rho, tau_fast, amp_adapt, gap = _params
    u, s, h = (float(row[k]) for k in ('u_now', 's_grad', 'H'))
    fast = -math.expm1(-h / tau_fast)
    slow = -math.expm1(-h / (tau_fast + gap))
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * fast - amp_adapt * u * slow)
