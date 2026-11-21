import math
import numpy as np
from scipy.optimize import least_squares

_COEF = None

def _features(row):
    return np.array([1.0, math.log1p(float(row['u_1'])), float(row['u_2'])])

def fit(train_rows):
    global _COEF
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    result = least_squares(lambda c: X @ c - y, np.zeros(X.shape[1]))
    _COEF = result.x
    return _COEF

def predict(row):
    if _COEF is None:
        raise RuntimeError('fit must be called before predict')
    return float(_features(row) @ _COEF)
