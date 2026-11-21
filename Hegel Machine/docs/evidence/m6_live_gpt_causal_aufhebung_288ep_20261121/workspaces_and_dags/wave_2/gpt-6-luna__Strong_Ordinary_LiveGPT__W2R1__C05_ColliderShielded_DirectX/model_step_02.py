import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return np.array([x / (1.1 + x), u], dtype=float)

def fit(train_rows):
    global _PARAMS
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda p: X @ p - y, x0=np.array([1.0, 1.0]))
    _PARAMS = result.x
    return _PARAMS.copy()

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit must be called before predict')
    return float(_features(row) @ _PARAMS)
