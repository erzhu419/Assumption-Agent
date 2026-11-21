import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    return np.array([float(row['u_now']), 1.0])

def fit(train_rows):
    global _PARAMS
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    result = least_squares(lambda p: X @ p - y, x0=np.array([2.0, 0.0]))
    _PARAMS = result.x
    return _PARAMS.copy()

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_features(row) @ _PARAMS)
