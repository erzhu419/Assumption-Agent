import math
import numpy as np
from scipy.optimize import least_squares

_COEF = None

def _features(row):
    u1 = float(row['u_1'])
    u2 = float(row['u_2'])
    return np.array([math.log1p(u1), u2 / (1.0 + u2)], dtype=float)

def fit(train_rows):
    global _COEF
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda c: X @ c - y, x0=np.zeros(2))
    _COEF = result.x
    return _COEF

def predict(row) -> float:
    if _COEF is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_features(row) @ _COEF)
