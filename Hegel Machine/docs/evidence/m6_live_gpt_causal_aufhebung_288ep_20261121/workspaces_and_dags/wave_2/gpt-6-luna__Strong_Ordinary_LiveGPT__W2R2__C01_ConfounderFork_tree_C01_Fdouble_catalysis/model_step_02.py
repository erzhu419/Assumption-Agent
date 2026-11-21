import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, key, default=0.0):
    return float(row.get(key, default))

def _model(row, p):
    vmax, k_half, c_ctrl = p
    x = _value(row, 'x_ind') / 1.5
    x2 = x * x
    return vmax * x2 / (k_half + x2) - c_ctrl * _value(row, 'u_ctrl')

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([_value(r, 'y') for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([7.0, 1.0, 0.2]), bounds=([0.0, 1e-8, 0.0], [np.inf, np.inf, np.inf]))
    _params = result.x
    return tuple(float(v) for v in _params)

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_model(row, _params))