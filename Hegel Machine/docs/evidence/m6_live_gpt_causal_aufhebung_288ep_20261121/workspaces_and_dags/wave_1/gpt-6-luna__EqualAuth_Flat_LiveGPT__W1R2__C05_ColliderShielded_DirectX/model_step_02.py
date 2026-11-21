import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    x = np.asarray([float(r['x_ind']) for r in train_rows], dtype=float)
    u = np.asarray([float(r['u_ctrl']) for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        vmax, c_ctrl = p
        return vmax * x / (1.1 + x) + c_ctrl * u - y
    result = least_squares(residual, x0=np.array([5.0, 1.0]))
    _PARAMS = (float(result.x[0]), float(result.x[1]))
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    vmax, c_ctrl = _PARAMS
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return float(vmax * x / (1.1 + x) + c_ctrl * u)
