import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    return float(row['u_now']), float(row['s_grad'])

def _model(p, row):
    u, s = _features(row)
    # Saturating response to present input, with a gradient-dependent offset
    # and input interaction to capture the observed branch inhibition.
    a, k, b, c = p
    base = a * (1.0 - math.exp(-k * max(u, 0.0)))
    return base + b * (s - 1.0) + c * (s - 1.0) * u

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_model(p, r) for r in rows]) - y
    starts = [np.array([2.8, 1.0, 0.0, 0.0]), np.array([3.0, 0.5, 0.0, 0.0]), np.array([2.5, 2.0, 0.0, 0.0])]
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    _PARAMS = best.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(_PARAMS, row))