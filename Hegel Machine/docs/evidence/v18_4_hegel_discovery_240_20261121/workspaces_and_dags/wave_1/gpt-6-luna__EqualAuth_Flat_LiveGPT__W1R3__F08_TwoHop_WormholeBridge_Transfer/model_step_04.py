import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return np.array([1.0, u, u*u, s, s*s, u*s], dtype=float)

def fit(train_rows):
    global _params
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    result = least_squares(lambda p: X @ p - y, np.zeros(X.shape[1]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_features(row) @ _params)
