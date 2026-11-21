import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(p, u, s, h, expm1):
    vmax, rho, adaptation, tau_fast, tau_slow = p
    siphon = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    fast = -expm1(-h / tau_fast)
    slow = -expm1(-h / tau_slow)
    return siphon * fast - adaptation * u * slow

def fit(train_rows):
    global _params
    u, s, h, y = np.asarray([[r['u_now'], r['s_grad'], r['H'], r['y']] for r in train_rows], dtype=float).T
    def residual(p):
        return _response(p, u, s, h, np.expm1) - y
    result = least_squares(residual, [6.0, 0.65, 1.0, 0.5, 10.0], bounds=([0.0, 0.0, 0.0, 1e-6, 1e-6], [np.inf, 1.0, np.inf, np.inf, np.inf]), ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row) -> float:
    return float(_response(_params, row['u_now'], row['s_grad'], row['H'], math.expm1))