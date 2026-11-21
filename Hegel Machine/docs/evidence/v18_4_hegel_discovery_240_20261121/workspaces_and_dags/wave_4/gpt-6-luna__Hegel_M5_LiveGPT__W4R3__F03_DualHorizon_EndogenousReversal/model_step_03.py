import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _law(row, p):
    a_fast, a_slow, tau_fast, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (a_fast * (-math.expm1(-h / tau_fast)) - a_slow * (-math.expm1(-h / tau_slow)))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    ys = np.array([float(r['y']) for r in rows])
    def residual(z):
        p = np.exp(z)
        return np.array([_law(r, p) for r in rows]) - ys
    starts = [
        np.log([1.0, 0.8, 0.2, 12.0]),
        np.log([1.2, 0.7, 0.3, 20.0]),
        np.log([0.8, 0.6, 0.1, 8.0]),
        np.log([1.5, 1.0, 0.5, 30.0]),
    ]
    best = None
    for z0 in starts:
        result = least_squares(residual, z0, max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _PARAMS = np.exp(best[1])
    return predict

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) first')
    return float(_law(row, _PARAMS))