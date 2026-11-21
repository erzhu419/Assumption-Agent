import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_TAU = 4.0

def _features(row):
    un = float(row['u_now']) / _TAU
    s = float(row['s_grad'])
    return un, s

def _predict_params(row, p):
    vmax, rho, c_loc = p
    un, s = _features(row)
    return vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un)) + c_loc * math.sqrt(max(0.0, un))

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in train_rows]) - y
    p0 = np.array([6.5, 0.70, 0.15])
    result = least_squares(residual, p0, bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    _PARAMS = result.x
    return {'vmax': float(_PARAMS[0]), 'rho': float(_PARAMS[1]), 'c_loc': float(_PARAMS[2]), 'tau_scale': _TAU}

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_params(row, _PARAMS))