import math
import numpy as np
from scipy.optimize import least_squares

_TAU = 0.005
_PARAMS = None

def _features(row):
    un = float(row['u_now']) / _TAU
    s = float(row['s_grad'])
    base = (1.0 - 0.70 * s / (1.0 + s)) * (un / (1.2 + un))
    return un, s, base

def _value(row, p):
    un, s, base = _features(row)
    vmax, rho, c_loc = p
    base = (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))
    return vmax * base + c_loc * math.sqrt(max(un, 0.0))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in rows], dtype=float) - y
    result = least_squares(residual, x0=np.array([6.5, 0.70, 0.15]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    _PARAMS = result.x.copy()
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _PARAMS))