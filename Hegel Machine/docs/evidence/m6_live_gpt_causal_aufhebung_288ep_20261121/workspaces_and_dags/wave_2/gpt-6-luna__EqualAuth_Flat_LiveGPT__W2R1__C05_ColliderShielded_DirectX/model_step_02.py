import numpy as np
from scipy.optimize import least_squares

_params = np.array([0.0, 0.0])

def fit(train_rows):
    global _params
    x = np.array([float(r['x_ind']) for r in train_rows])
    u = np.array([float(r['u_ctrl']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    A = np.column_stack((x / (1.1 + x), u))
    result = least_squares(lambda p: A @ p - y, x0=np.array([1.0, 1.0]))
    _params = result.x
    return _params

def predict(row):
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return float(_params[0] * x / (1.1 + x) + _params[1] * u)
