import math
import numpy as np
from scipy.optimize import least_squares

_params = np.array([1.0, 1.0], dtype=float)

def fit(train_rows):
    global _params
    x = np.array([float(r['x_ind']) for r in train_rows], dtype=float)
    u = np.array([float(r['u_ctrl']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    h = x / (1.1 + x)
    result = least_squares(lambda p: p[0] * h + p[1] * u - y, _params, loss='linear')
    _params = result.x
    return predict

def predict(row):
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return float(_params[0] * x / (1.1 + x) + _params[1] * u)
