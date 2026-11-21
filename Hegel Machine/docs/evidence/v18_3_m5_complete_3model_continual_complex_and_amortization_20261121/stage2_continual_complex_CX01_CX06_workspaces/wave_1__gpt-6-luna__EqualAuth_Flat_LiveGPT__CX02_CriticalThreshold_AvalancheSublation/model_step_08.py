import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return np.array([1.0, u, s, u*u, u*s, s*s, u**3, u*u*s, u*s*s, s**3], dtype=float)

def fit(train_rows):
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    scale = np.maximum(np.linalg.norm(X, axis=0), 1.0)
    Z = X / scale
    def residual(beta):
        return Z @ beta - y
    result = least_squares(residual, np.zeros(X.shape[1]))
    return result.x / scale

def predict(row):
    return float(_features(row) @ _COEFFICIENTS)
