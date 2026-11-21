import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = (0.0, 0.0)

def fit(train_rows):
    global _PARAMS
    x = np.asarray([float(r['x_ind']) for r in train_rows], dtype=float)
    u = np.asarray([float(r['u_ctrl']) for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    f = x / (1.1 + x)
    result = least_squares(lambda p: p[0] * f + p[1] * u - y,
                           x0=np.array([max(y) * 1.5, 0.0]))
    _PARAMS = (float(result.x[0]), float(result.x[1]))
    return _PARAMS

def predict(row):
    vmax, c_ctrl = _PARAMS
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return float(vmax * x / (1.1 + x) + c_ctrl * u)
