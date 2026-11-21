import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    X = np.array([[r['u_now'], r['x_now'], r['H'], 1.0] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    _params = least_squares(lambda p: X @ p - y, np.array([1.75, 0.0, 0.0, 0.0])).x

def predict(row):
    return float(np.dot(_params, [row['u_now'], row['x_now'], row['H'], 1.0]))