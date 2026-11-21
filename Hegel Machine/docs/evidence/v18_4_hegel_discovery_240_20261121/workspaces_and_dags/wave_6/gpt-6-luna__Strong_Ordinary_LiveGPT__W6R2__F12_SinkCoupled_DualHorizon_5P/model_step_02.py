import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(p, row):
    vmax, rho, amp_adapt, tau_fast, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base * (1.0 - math.exp(-h / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-h / tau_slow))

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    lo = np.array([0.0, 0.0, 0.0, 1e-4, 0.5])
    hi = np.array([20.0, 0.999, 10.0, 10.0, 200.0])
    def residual(p):
        return np.asarray([_value(p, r) for r in train_rows]) - y
    starts = [
        [4.0, 0.5, 1.0, 0.5, 20.0],
        [6.0, 0.3, 1.5, 1.0, 30.0],
        [3.0, 0.7, 0.8, 0.2, 15.0],
        [8.0, 0.6, 2.0, 2.0, 50.0],
    ]
    best = None
    for start in starts:
        result = least_squares(residual, np.clip(start, lo + 1e-8, hi - 1e-8), bounds=(lo, hi), max_nfev=20000)
        score = float(np.sum(result.fun ** 2))
        if best is None or score < best[0]:
            best = (score, result.x)
    _PARAMS = best[1]
    return _PARAMS.copy()

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_value(_PARAMS, row))