import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base * (1.0 - math.exp(-h / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-h / tau_slow))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    lo = np.array([1e-6, 0.0, 1e-4, 1e-6, 1e-3])
    hi = np.array([100.0, 0.999, 100.0, 100.0, 1e4])
    starts = [
        [3.0, 0.4, 0.3, 1.0, 20.0],
        [5.0, 0.7, 0.1, 1.5, 10.0],
        [2.0, 0.2, 1.0, 0.8, 40.0],
        [10.0, 0.8, 0.5, 2.0, 25.0]
    ]
    sols = [least_squares(residual, np.clip(x, lo + 1e-9, hi - 1e-9), bounds=(lo, hi), max_nfev=20000) for x in starts]
    _params = min(sols, key=lambda z: float(np.dot(z.fun, z.fun))).x
    return _params.copy()

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_value(row, _params))