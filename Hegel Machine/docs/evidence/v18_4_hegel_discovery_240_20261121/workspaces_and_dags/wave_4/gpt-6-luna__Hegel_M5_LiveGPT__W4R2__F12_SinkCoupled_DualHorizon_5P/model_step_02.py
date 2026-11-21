import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    sink = 1.0 - rho * s / (1.0 + s)
    base = vmax * sink * (u / (1.2 + u)) * (1.0 - math.exp(-h / tau_fast))
    adapt = amp_adapt * u * (1.0 - math.exp(-h / tau_slow))
    return base - adapt

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    starts = [
        [6.0, 0.70, 0.5, 0.10, 10.0],
        [6.0, 0.70, 1.0, 0.15, 20.0],
        [5.0, 0.65, 0.3, 0.08, 8.0],
    ]
    fits = [least_squares(residual, x0, bounds=([0.01, 0.0, 0.01, 0.0, 0.1], [30.0, 0.999, 10.0, 5.0, 200.0]), max_nfev=20000) for x0 in starts]
    best = min(fits, key=lambda z: float(np.sum(z.fun ** 2)))
    _PARAMS = best.x.copy()
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _PARAMS))