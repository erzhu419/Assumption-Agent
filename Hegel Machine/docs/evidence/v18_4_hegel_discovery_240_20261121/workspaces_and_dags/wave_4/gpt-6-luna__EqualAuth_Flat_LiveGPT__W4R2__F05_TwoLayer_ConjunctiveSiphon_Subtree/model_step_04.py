import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u = max(float(row['u_now']), 1e-8)
    s = max(float(row['s_grad']), 0.0)
    # Two smooth production layers with sink-gradient attenuation.
    return u, s

def _surface(p, row):
    a, b, c, d, e = p
    u, s = _features(row)
    base = a * u / (b + u) + c * math.sqrt(u)
    attenuation = 1.0 - d * s / (1.0 + s)
    return base * attenuation + e * u

def fit(train_rows):
    global _PARAMS
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_surface(p, r) for r in train_rows]) - ys
    starts = [np.array([20., .3, 10., .8, 1.]), np.array([10., .1, 15., .5, 0.])]
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=([0., 1e-5, 0., 0., -100.], [500., 10., 100., .999, 100.]), max_nfev=20000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _PARAMS = best.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_surface(_PARAMS, row))