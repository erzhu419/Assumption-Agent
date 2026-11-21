import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return np.array([1.0, u, u*u, s, s*s, u*s], dtype=float)

def fit(train_rows):
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return X @ p - y
    result = least_squares(residual, np.zeros(X.shape[1]), max_nfev=5000)
    return result.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_features(row) @ p)