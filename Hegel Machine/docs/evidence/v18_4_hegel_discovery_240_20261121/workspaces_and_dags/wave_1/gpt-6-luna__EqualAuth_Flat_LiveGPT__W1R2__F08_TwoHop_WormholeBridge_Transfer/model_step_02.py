import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now'])
    g = float(row['s_grad']) - 1.0
    return np.array([1.0, u, u*u, g, g*u], dtype=float)

def fit(train_rows):
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    res = least_squares(lambda p: X @ p - y, np.zeros(X.shape[1]))
    return res.x

def predict(row):
    return float(_features(row) @ _PARAMS)

# The harness fits first; retain fitted coefficients for predict.
_PARAMS = np.zeros(5)
_FIT = fit
def fit(train_rows):
    global _PARAMS
    _PARAMS = _FIT(train_rows)
    return _PARAMS
