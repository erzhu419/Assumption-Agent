import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u = max(0.0, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    # Smooth low-order response surface; inputs are scaled to the observed range.
    a, b = u / 0.2, s / 2.5
    return np.array([1.0, a, a*a, b, b*b, a*b, a*a*b, a*b*b], dtype=float)

def fit(train_rows):
    global _PARAMS
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    # Mild ridge regularization stabilizes the surface without materially
    # constraining the observed response.
    def residual(p):
        return np.r_[X @ p - y, 0.015 * p[1:]]
    result = least_squares(residual, np.zeros(X.shape[1]), max_nfev=10000)
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_features(row) @ _PARAMS)
