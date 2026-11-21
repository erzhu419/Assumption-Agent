import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    X = np.array([[r['u_now'], r['x_now']] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    result = least_squares(lambda p: X @ p - y, np.array([1.0, 0.0]))
    _params = result.x
    return _params

def predict(row):
    return float(_params[0] * row['u_now'] + _params[1] * row['x_now'])
