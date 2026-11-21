import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, amp_adapt, tau_fast, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base * (1.0 - math.exp(-h / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-h / tau_slow))

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    starts = [
        [3.0, 0.7, 0.8, 0.5, 10.0],
        [5.0, 0.5, 0.6, 1.0, 20.0],
        [2.0, 0.9, 1.0, 0.2, 30.0],
    ]
    fits = [least_squares(residual, x0, bounds=([0.0, 0.0, 0.0, 1e-4, 0.5], [100.0, 1.0, 20.0, 20.0, 500.0]), max_nfev=20000) for x0 in starts]
    _params = min(fits, key=lambda z: float(np.dot(z.fun, z.fun))).x
    return _params

def predict(row) -> float:
    if _params is None:
        raise ValueError('predict requires fitted parameters; call fit(train_rows) first')
    return float(_model(row, _params))