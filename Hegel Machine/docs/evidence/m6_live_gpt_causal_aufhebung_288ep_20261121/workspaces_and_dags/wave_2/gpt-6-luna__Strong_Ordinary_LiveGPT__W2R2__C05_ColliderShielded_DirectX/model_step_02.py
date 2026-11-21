import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _scalar(value):
    arr = np.asarray(value, dtype=float).reshape(-1)
    return float(arr[0])

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    x = np.array([_scalar(r['x_ind']) for r in rows], dtype=float)
    u = np.array([_scalar(r['u_ctrl']) for r in rows], dtype=float)
    y = np.array([_scalar(r['y']) for r in rows], dtype=float)
    def residual(p):
        vmax, c_ctrl = p
        return vmax * x / (1.1 + x) + c_ctrl * u - y
    result = least_squares(residual, x0=np.array([8.0, 0.5]))
    _PARAMS = tuple(float(v) for v in result.x)
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    vmax, c_ctrl = _PARAMS
    x = _scalar(row['x_ind'])
    u = _scalar(row['u_ctrl'])
    return float(vmax * x / (1.1 + x) + c_ctrl * u)
