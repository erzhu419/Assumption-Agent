import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(p, row):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (1.0 - math.exp(-h / tau_fast))
            - amp_adapt * u * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(p, r) for r in rows]) - y
    starts = [np.array([4.0, 0.5, 1.0, 0.15, 12.0]),
              np.array([3.0, 0.3, 0.5, 0.1, 20.0]),
              np.array([5.0, 0.8, 2.0, 0.2, 8.0])]
    best = None
    for x0 in starts:
        result = least_squares(residual, x0,
            bounds=([0.0, 0.0, 1e-4, 0.0, 1e-3], [20.0, 2.0, 100.0, 5.0, 500.0]),
            max_nfev=20000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _PARAMS = best.x.copy()
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(_PARAMS, row))