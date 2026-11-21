import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    x = np.array([float(r['x_ind']) for r in train_rows], dtype=float)
    u = np.array([float(r['u_ctrl']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    h = x / (1.1 + x)
    result = least_squares(lambda p: p[0] * h + p[1] * u - y, x0=np.array([1.0, 1.0]))
    _PARAMS = result.x
    return _PARAMS.copy()

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_PARAMS[0] * float(row['x_ind']) / (1.1 + float(row['x_ind'])) + _PARAMS[1] * float(row['u_ctrl']))