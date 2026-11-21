import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, amp_adapt, tau_fast, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base * (1.0 - math.exp(-H / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-H / tau_slow))

def fit(train_rows):
    global _params
    ys = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - ys
    lower = np.array([0.0, 0.0, 0.0, 0.01, 0.02])
    upper = np.array([20.0, 0.999, 10.0, 20.0, 100.0])
    starts = [
        [3.0, 0.7, 1.0, 0.5, 5.0],
        [4.0, 0.5, 1.0, 1.0, 10.0],
        [2.0, 0.8, 0.5, 0.2, 20.0],
        [5.0, 0.3, 1.5, 2.0, 30.0],
    ]
    best = None
    for start in starts:
        result = least_squares(residual, np.clip(start, lower + 1e-8, upper - 1e-8), bounds=(lower, upper), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _params = best[1]
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))