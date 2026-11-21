import numpy as np
from scipy.optimize import least_squares

_P = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return np.array([1.0, u, s, u*u, u*s, s*s], dtype=float)

def fit(train_rows):
    global _P
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda p: X @ p - y, np.zeros(X.shape[1]))
    _P = result.x
    return _P.copy()

def predict(row) -> float:
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_features(row) @ _P)
