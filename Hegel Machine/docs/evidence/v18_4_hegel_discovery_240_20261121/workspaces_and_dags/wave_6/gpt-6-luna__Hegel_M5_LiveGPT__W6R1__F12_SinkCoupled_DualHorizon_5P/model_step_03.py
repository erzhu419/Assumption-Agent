import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _law(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            * (1.0 - math.exp(-h / tau_fast))
            - amp_adapt * u * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _PARAMS
    ys = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_law(r, p) for r in train_rows]) - ys
    starts = [
        [6.0, 0.70, 0.35, 0.8, 10.0],
        [5.0, 0.65, 0.8, 0.5, 20.0],
        [8.0, 0.75, 0.15, 1.0, 6.0],
    ]
    lo = [0.01, 0.0, 0.01, 0.0, 0.05]
    hi = [30.0, 0.999, 10.0, 10.0, 200.0]
    best = None
    for x0 in starts:
        res = least_squares(residual, x0, bounds=(lo, hi), max_nfev=30000)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    _PARAMS = best.x
    return _PARAMS.copy()

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_law(row, _PARAMS))
