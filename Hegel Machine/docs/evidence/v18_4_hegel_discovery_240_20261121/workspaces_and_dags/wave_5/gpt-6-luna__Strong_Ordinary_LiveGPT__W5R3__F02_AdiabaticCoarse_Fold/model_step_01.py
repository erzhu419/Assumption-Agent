import numpy as np
from scipy.optimize import least_squares

def _features(row):
    return np.array([float(row['u_now']), 1.0], dtype=float)

def fit(train_rows):
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    p0 = np.linalg.lstsq(X, y, rcond=None)[0]
    res = least_squares(lambda p: X @ p - y, p0)
    return res.x

def predict(row):
    return float(_features(row) @ fit(train_rows))