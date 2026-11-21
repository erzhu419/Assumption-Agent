import math
import numpy as np
from scipy.optimize import least_squares

_params = {'vmax': 1.0, 'c_ctrl': 1.0}

def fit(train_rows):
    global _params
    x = np.asarray([float(r['x_ind']) for r in train_rows], dtype=float)
    u = np.asarray([float(r['u_ctrl']) for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    h = x / (1.1 + x)
    result = least_squares(lambda p: p[0] * h + p[1] * u - y, x0=np.array([1.0, 1.0]))
    _params = {'vmax': float(result.x[0]), 'c_ctrl': float(result.x[1])}
    return _params

def predict(row):
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return float(_params['vmax'] * x / (1.1 + x) + _params['c_ctrl'] * u)
