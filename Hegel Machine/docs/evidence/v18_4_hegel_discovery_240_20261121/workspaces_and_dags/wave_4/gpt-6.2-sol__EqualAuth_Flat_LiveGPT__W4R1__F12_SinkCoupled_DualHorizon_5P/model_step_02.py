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
        vmax, rho, amp, fast, gap = p
        slow = fast + gap
        return vmax * (1.0 - rho * sink) * drive * (-np.expm1(-h / fast)) - amp * u * (-np.expm1(-h / slow)) - y

    result = least_squares(residual, [5.0, 0.5, 1.0, 0.5, 9.5], bounds=([0.0, 0.0, 0.0, 1e-6, 1e-6], [np.inf, 1.0, np.inf, np.inf, np.inf]), x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12)
    vmax, rho, amp, fast, gap = map(float, result.x)
    _params = (vmax, rho, amp, fast, fast + gap)
    return _params

def predict(row) -> float:
    vmax, rho, amp, fast, slow = _params
    u, s, h = (float(row[k]) for k in ('u_now', 's_grad', 'H'))
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (-math.expm1(-h / fast)) - amp * u * (-math.expm1(-h / slow))
