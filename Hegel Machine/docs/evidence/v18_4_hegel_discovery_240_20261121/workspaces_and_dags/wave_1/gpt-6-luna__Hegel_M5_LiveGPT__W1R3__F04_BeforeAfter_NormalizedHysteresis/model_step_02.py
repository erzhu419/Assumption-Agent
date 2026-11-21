import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    return float(row['x_now']), float(row['u_now']) - float(row['u_past']), float(row['H'])

def fit(train_rows):
    global _PARAMS
    X = np.array([_features(r) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        a, k = p
        return X[:, 0] + a * X[:, 1] * np.exp(-k * X[:, 2]) - y
    result = least_squares(residual, np.array([500.0, 0.4]))
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    a, k = _PARAMS
    x, delta, h = _features(row)
    return float(x + a * delta * math.exp(-k * h))