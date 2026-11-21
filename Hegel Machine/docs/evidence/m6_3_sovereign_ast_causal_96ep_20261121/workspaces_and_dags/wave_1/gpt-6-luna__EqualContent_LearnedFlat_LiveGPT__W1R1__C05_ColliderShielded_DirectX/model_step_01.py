import numpy as np
from scipy.optimize import least_squares

_params = np.array([1.0, 1.0], dtype=float)

def fit(train_rows):
    global _params
    X = np.array([[float(r['x_ind'])/(1.1+float(r['x_ind'])), float(r['u_ctrl'])] for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    res = least_squares(lambda p: X @ p - y, x0=np.array([1.0, 1.0]))
    _params = res.x
    return predict

def predict(row):
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return float(_params[0] * x / (1.1 + x) + _params[1] * u)
