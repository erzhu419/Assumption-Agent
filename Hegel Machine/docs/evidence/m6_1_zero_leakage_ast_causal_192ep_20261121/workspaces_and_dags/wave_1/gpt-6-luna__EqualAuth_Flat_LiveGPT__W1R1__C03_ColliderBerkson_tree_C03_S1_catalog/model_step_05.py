import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    x = float(row['u_1'])
    z = float(row['u_2'])
    return np.array([1.0, math.log1p(x), math.log1p(z), x*z, x*x, z*z], dtype=float)

def fit(train_rows):
    global _params
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    res = least_squares(lambda p: X @ p - y, np.zeros(X.shape[1]))
    _params = res.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) first.')
    return float(_features(row) @ _params)