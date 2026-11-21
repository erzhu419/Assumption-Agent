import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base * (1.0 - math.exp(-H / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-H / tau_slow))

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    starts = [
        [2.5, 0.35, 1.0, 0.8, 20.0],
        [4.0, 0.6, 0.4, 1.0, 10.0],
        [2.0, 0.2, 2.0, 0.5, 40.0],
    ]
    fits = [least_squares(residual, x0, bounds=([0.0, 0.0, 1e-4, 0.0, 1e-3], [100.0, 1.0, 100.0, 100.0, 1e4]), max_nfev=20000) for x0 in starts]
    _params = min(fits, key=lambda f: float(np.dot(f.fun, f.fun))).x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _params))