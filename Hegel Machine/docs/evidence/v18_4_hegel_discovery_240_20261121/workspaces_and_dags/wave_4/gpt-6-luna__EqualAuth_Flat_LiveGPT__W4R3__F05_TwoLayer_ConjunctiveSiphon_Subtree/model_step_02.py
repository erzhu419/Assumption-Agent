import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    return np.array([1.0, u, np.sqrt(u), s, u*s], dtype=float)

def fit(train_rows):
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return X @ p - y
    result = least_squares(residual, np.zeros(X.shape[1]))
    p = result.x
    def predict(row):
        return float(_features(row) @ p)
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) first to obtain a fitted predictor.')