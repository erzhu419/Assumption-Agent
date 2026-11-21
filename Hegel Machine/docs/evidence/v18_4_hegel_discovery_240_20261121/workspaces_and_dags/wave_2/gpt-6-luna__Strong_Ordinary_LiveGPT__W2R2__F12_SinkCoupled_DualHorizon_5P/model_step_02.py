import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_with(p, row):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    sink = 1.0 - rho * s / (1.0 + s)
    base = vmax * sink * (u / (1.2 + u)) * (-math.expm1(-H / tau_fast))
    adapt = amp_adapt * u * (-math.expm1(-H / tau_slow))
    return base - adapt

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_with(p, r) for r in rows]) - y
    starts = [
        [3.0, 0.7, 0.25, 0.9, 12.0],
        [4.0, 0.5, 0.5, 0.7, 20.0],
        [2.5, 1.0, 0.15, 1.0, 8.0],
        [5.0, 0.8, 1.0, 0.6, 30.0],
    ]
    best = None
    for x0 in starts:
        res = least_squares(residual, x0=x0,
            bounds=([0.0, 0.0, 1e-5, 0.0, 1e-5], [100.0, 5.0, 100.0, 100.0, 1000.0]),
            max_nfev=20000)
        score = float(np.dot(res.fun, res.fun))
        if best is None or score < best[0]:
            best = (score, res.x)
    _PARAMS = best[1]
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_predict_with(_PARAMS, row))