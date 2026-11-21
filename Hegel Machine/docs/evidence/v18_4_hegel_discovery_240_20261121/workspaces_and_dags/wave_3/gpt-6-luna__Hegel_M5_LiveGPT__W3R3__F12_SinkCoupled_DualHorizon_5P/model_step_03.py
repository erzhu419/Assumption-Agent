import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            * (1.0 - math.exp(-h / tau_fast))
            - amp_adapt * u * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    starts = [
        [6.5, 0.70, 0.8, 0.8, 12.0],
        [6.5, 0.70, 2.0, 0.8, 20.0],
        [5.0, 0.65, 0.3, 0.5, 8.0],
        [8.0, 0.75, 3.0, 1.0, 30.0],
    ]
    lo = [0.01, 0.01, 0.01, 0.0, 0.05]
    hi = [30.0, 0.99, 100.0, 10.0, 200.0]
    fits = [least_squares(residual, x0, bounds=(lo, hi), max_nfev=20000) for x0 in starts]
    _params = min(fits, key=lambda z: np.sum(z.fun ** 2)).x
    return _params.copy()

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) first.')
    return float(_value(row, _params))