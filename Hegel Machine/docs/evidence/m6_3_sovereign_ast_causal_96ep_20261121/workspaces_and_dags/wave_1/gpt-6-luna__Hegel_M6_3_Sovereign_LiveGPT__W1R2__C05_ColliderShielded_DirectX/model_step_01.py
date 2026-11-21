import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    x = np.array([float(r['x_ind']) for r in train_rows])
    u = np.array([float(r['u_ctrl']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def pred(p):
        vmax, c = p
        return vmax * x / (1.1 + x) + c * u
    res = least_squares(lambda p: pred(p) - y, x0=[5.0, 0.5])
    _params = tuple(float(v) for v in res.x)
    return lambda row: float(_params[0] * float(row['x_ind']) / (1.1 + float(row['x_ind'])) + _params[1] * float(row['u_ctrl']))

def predict(row):
    if _params is None:
        raise RuntimeError('fit must be called first')
    return float(_params[0] * float(row['x_ind']) / (1.1 + float(row['x_ind'])) + _params[1] * float(row['u_ctrl']))