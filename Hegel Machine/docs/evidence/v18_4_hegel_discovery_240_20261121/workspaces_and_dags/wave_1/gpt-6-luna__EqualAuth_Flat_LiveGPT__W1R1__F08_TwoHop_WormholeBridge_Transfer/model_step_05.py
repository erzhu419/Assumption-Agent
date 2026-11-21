import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return u, s

def _model(p, row):
    u, s = _features(row)
    # Saturating baseline plus a gradient-dependent polynomial modulation.
    base = p[0] + p[1] * u / (p[2] + u)
    d = s - 1.0
    mod = d * (p[3] + p[4]*u + p[5]*u*u) + d*d * (p[6] + p[7]*u)
    return base + mod

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_model(p, r) for r in train_rows]) - y
    starts = [np.array([0.0, 6.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
              np.array([1.0, 5.0, 1.0, 1.0, -1.0, 0.1, 0.1, 0.0])]
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, max_nfev=30000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    _PARAMS = best.x
    return _PARAMS.copy()

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('call fit(train_rows) before predict(row)')
    return float(_model(_PARAMS, row))