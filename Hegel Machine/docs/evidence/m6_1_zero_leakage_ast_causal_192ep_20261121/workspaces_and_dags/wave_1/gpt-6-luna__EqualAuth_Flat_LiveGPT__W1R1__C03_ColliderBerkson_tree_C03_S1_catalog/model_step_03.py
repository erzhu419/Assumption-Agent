import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u1 = float(row['u_1'])
    u2 = float(row['u_2'])
    return math.log1p(u1), u2 / (1.5 + u2)

def fit(train_rows):
    global _PARAMS
    X = np.array([[1.0, *_features(r)] for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda p: X @ p - y, x0=np.array([0.0, 3.0, 4.0]))
    _PARAMS = result.x
    return predict

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) first.')
    return float(np.array([1.0, *_features(row)]) @ _PARAMS)
