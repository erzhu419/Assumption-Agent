import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    X = np.array([[r['u_now'], r['x_now'], 1.0] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    _params = least_squares(lambda p: X @ p - y, np.array([2.5, 0.0, 0.0])).x

def predict(row):
    return float(_params[0] * row['u_now'] + _params[1] * row['x_now'] + _params[2])