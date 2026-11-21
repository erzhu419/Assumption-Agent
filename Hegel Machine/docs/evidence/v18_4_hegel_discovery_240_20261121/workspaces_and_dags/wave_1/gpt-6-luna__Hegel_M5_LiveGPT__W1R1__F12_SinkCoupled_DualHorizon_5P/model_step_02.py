import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(p, row):
    vmax, rho, amp_adapt, tau_fast, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base * (1.0 - math.exp(-H / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-H / tau_slow))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_predict_with(p, r) for r in rows]) - y
    lower = [0.0, 0.0, 0.0, 1e-4, 1e-3]
    upper = [30.0, 0.999999, 10.0, 100.0, 500.0]
    starts = [
        [6.0, 0.7, 0.8, 1.0, 20.0],
        [5.0, 0.5, 0.5, 0.5, 10.0],
        [8.0, 0.8, 1.0, 2.0, 40.0],
    ]
    best = None
    for x0 in starts:
        result = least_squares(residual, x0, bounds=(lower, upper), max_nfev=30000, xtol=1e-13, ftol=1e-13, gtol=1e-13)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _params = best[1]
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_with(_params, row))