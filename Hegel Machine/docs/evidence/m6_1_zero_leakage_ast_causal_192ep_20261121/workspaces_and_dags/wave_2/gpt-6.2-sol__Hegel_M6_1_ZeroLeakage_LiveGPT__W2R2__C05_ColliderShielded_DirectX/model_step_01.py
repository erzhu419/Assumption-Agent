import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    x = np.array([float(r['x_ind']) for r in rows])
    u = np.array([float(r['u_ctrl']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    s = x / (1.1 + x)
    _params = least_squares(lambda p: p[0] * s + p[1] * u - y, [1.0, 0.0]).x
    return _params

def predict(row) -> float:
    vmax, c_ctrl = _params
    x = float(row['x_ind'])
    return math.fsum((vmax * x / (1.1 + x), c_ctrl * float(row['u_ctrl'])))