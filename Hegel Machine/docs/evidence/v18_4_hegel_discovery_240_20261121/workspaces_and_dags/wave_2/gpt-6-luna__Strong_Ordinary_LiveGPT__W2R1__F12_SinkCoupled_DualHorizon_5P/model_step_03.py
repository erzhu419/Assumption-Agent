import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
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
        return np.asarray([_value(r, p) for r in train_rows]) - y
    starts = [
        [3.0, 0.5, 0.5, 1.0, 10.0],
        [2.0, 0.7, 1.0, 1.0, 20.0],
        [4.0, 0.3, 0.2, 0.5, 8.0],
        [3.0, 0.8, 2.0, 1.0, 30.0],
    ]
    lower = [0.0, 0.0, 1e-4, 0.0, 1e-4]
    upper = [100.0, 1.0, 100.0, 100.0, 1000.0]
    fits = [least_squares(residual, x0, bounds=(lower, upper), max_nfev=20000) for x0 in starts]
    best = min(fits, key=lambda z: float(np.sum(z.fun ** 2)))
    _params = best.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))