import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return np.array([1.0, u, u*u, u*u*u, s, s*u], dtype=float)

def fit(train_rows):
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda p: X @ p - y, np.zeros(X.shape[1]), max_nfev=10000)
    return result.x

def predict(row):
    return float(_features(row) @ fit(_TRAIN_ROWS))