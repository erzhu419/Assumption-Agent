import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = (8.0, 0.85)

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    if not rows:
        _PARAMS = (8.0, 0.85)
        return _PARAMS
    x = np.array([float(r['x_ind']) / (1.1 + float(r['x_ind'])) for r in rows])
    u = np.array([float(r['u_ctrl']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    result = least_squares(lambda p: p[0] * x + p[1] * u - y, np.array([8.0, 0.85]))
    _PARAMS = (float(result.x[0]), float(result.x[1]))
    return _PARAMS

def predict(row):
    vmax, c_ctrl = _PARAMS
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return vmax * x / (1.1 + x) + c_ctrl * u