import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u = float(row['u_now']) / 2.5
    s = float(row['s_grad'])
    return u, s

def _law(params, row):
    vmax, k_half, alpha_inh, c_leak = params
    u, s = _features(row)
    u2 = u * u
    return vmax * (u2 / (k_half + u2)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _PARAMS
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(params):
        return np.asarray([_law(params, r) for r in train_rows], dtype=float) - ys
    result = least_squares(residual, x0=np.array([4.0, 1.0, 0.2, 0.1]), bounds=([0.0, 1e-10, 0.0, 0.0], [100.0, 100.0, 100.0, 100.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_law(_PARAMS, row))
