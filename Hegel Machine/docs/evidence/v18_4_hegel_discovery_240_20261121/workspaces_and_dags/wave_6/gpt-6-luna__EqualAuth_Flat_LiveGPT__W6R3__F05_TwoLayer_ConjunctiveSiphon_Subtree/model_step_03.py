import numpy as np
from scipy.optimize import least_squares

_COEF = None
_U0 = 0.01

def _features(row):
    u = float(row['u_now']) / _U0
    s = float(row['s_grad'])
    return np.array([1.0, u, u*u, s, s*s, u*s], dtype=float)

def fit(train_rows):
    global _COEF
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda p: X @ p - y, np.zeros(X.shape[1]), max_nfev=10000)
    _COEF = result.x
    return _COEF.copy()

def predict(row):
    if _COEF is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_features(row) @ _COEF)
