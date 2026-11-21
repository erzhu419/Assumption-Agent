import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    x = np.asarray([float(r['x_ind']) for r in rows], dtype=float)
    u = np.asarray([float(r['u_ctrl']) for r in rows], dtype=float)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    basis = x / (1.1 + x)
    result = least_squares(lambda p: p[0] * basis + p[1] * u - y,
                           np.array([7.0, 1.0], dtype=float))
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return float(_params[0] * x / (1.1 + x) + _params[1] * u)
