import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u = float(row['u_now'])
    g = float(row['s_grad'])
    return u, g

def _value(p, row):
    vmax, km, rho, leak, gcrit, power = p
    u, g = _features(row)
    un = u  # supplied characteristic tau_scale is 1.0
    pool = vmax * (1.0 - rho * g / (1.0 + g)) * un / (km + un)
    critical_leak = leak * max(g - gcrit, 0.0) ** power * un
    return pool - critical_leak

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(p, r) for r in rows]) - y
    starts = [
        [5.0, 1.2, 0.7, 1.0, 1.0, 1.5],
        [5.0, 1.2, 0.5, 2.0, 1.2, 2.0],
        [8.0, 2.0, 0.8, 0.5, 0.8, 1.0],
    ]
    lo = [0.01, 0.01, 0.0, 0.0, 0.0, 0.3]
    hi = [30.0, 20.0, 5.0, 30.0, 3.0, 4.0]
    best = None
    for start in starts:
        res = least_squares(residual, start, bounds=(lo, hi), max_nfev=20000)
        score = float(np.dot(res.fun, res.fun))
        if best is None or score < best[0]:
            best = (score, res.x)
    _PARAMS = best[1]
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(_PARAMS, row))
