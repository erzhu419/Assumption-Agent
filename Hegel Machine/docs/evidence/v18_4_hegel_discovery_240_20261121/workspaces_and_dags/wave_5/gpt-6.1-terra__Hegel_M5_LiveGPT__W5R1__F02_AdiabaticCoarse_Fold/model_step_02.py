import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now'])
    x = float(row['x_now'])
    h = float(row['H']) - 4.28
    return np.array([u, x, 1.0, u*h, x*h])

def fit(train_rows):
    global _params
    X = np.array([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.concatenate((X @ p - y, 0.01 * p[3:]))
    _params = least_squares(residual, np.array([2.24, 0.0, 0.0, 0.0, 0.0])).x

def predict(row):
    return float(_features(row) @ _params)
