import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    return np.array([float(row['u_now']), 1.0], dtype=float)

def fit(train_rows):
    global _params
    X = np.vstack([_features(row) for row in train_rows])
    y = np.array([float(row['y']) for row in train_rows], dtype=float)
    result = least_squares(lambda p: X @ p - y, x0=np.array([2.0, 0.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_features(row) @ _params)
