import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now'])
    return np.array([u / (1.2 + u), np.sqrt(max(u, 0.0))], dtype=float)

def fit(train_rows):
    global _params
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    res = least_squares(lambda p: X @ p - y, x0=np.array([1.0, 0.1]))
    _params = res.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_features(row) @ _params)
