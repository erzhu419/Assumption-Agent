import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_with(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    sink = 1.0 - rho * s / (1.0 + s)
    fast = vmax * sink * (u / (1.2 + u)) * (1.0 - math.exp(-H / tau_fast))
    adapt = amp_adapt * u * (1.0 - math.exp(-H / tau_slow))
    return fast - adapt

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_with(r, p) for r in train_rows]) - y
    lower = np.array([0.0, 0.0, 0.01, 0.0, 0.5])
    upper = np.array([20.0, 1.0, 10.0, 10.0, 100.0])
    starts = [
        [4.0, 0.5, 0.5, 0.8, 6.0],
        [6.0, 0.3, 1.0, 1.0, 10.0],
        [3.0, 0.7, 0.25, 0.5, 3.0],
    ]
    fits = [least_squares(residual, np.asarray(p), bounds=(lower, upper), max_nfev=3000) for p in starts]
    best = min(fits, key=lambda result: float(np.dot(result.fun, result.fun)))
    _PARAMS = best.x
    return _PARAMS.copy()

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_predict_with(row, _PARAMS))