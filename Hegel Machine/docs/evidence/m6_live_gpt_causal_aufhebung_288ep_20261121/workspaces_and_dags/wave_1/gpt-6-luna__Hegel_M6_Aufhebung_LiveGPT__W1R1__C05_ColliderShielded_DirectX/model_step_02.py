import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    x = np.array([float(r['x_ind']) / (1.1 + float(r['x_ind'])) for r in train_rows], dtype=float)
    u = np.array([float(r['u_ctrl']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return p[0] * x + p[1] * u - y
    result = least_squares(residual, x0=np.array([5.0, 1.0]))
    _params = (float(result.x[0]), float(result.x[1]))
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    vmax, c_ctrl = _params
    x = float(row['x_ind'])
    return vmax * x / (1.1 + x) + c_ctrl * float(row['u_ctrl'])