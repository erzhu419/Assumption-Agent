import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (1.0 - math.exp(-h / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-h / tau_slow))

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_evaluate(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([4.0, 0.4, 1.0, 0.15, 12.0]), bounds=([0.0, 0.0, 1e-5, 0.0, 1e-5], [100.0, 1.0, 100.0, 10.0, 1000.0]), max_nfev=20000)
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_evaluate(row, _params))