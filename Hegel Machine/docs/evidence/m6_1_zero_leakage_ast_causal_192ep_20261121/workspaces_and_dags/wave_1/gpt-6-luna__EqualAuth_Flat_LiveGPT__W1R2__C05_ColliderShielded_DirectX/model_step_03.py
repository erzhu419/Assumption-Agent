import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    x = np.array([float(r['x_ind']) for r in train_rows], dtype=float)
    u = np.array([float(r['u_ctrl']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    basis = x / (1.1 + x)
    result = least_squares(lambda p: p[0] * basis + p[1] * u - y, x0=np.array([5.0, 0.5]))
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return float(_params[0] * x / (1.1 + x) + _params[1] * u)