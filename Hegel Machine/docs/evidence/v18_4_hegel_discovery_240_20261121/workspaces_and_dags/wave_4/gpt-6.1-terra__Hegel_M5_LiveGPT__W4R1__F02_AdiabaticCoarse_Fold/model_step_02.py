import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now'])
    return np.array([u, float(row['x_now']), u * (float(row['H']) - 3.45), 1.0])

def fit(train_rows):
    global _params
    X = np.array([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    _params = least_squares(lambda p: X @ p - y, np.array([1.95, 0.0, 0.0, 0.0])).x

def predict(row):
    return float(_features(row) @ _params)
