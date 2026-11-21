import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now'])
    g = float(row['s_grad'])
    q = (0.2 * u) ** 4
    return np.array([u, -q, g, g * u], dtype=float)

def fit(train_rows):
    global _params
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    # Numerically fit the quartic radiative-flux contribution and the
    # gradient-dependent bridge offset and input coupling.
    res = least_squares(lambda p: X @ p - y,
                        x0=np.array([1.0, 1.0, 0.0, 0.0]),
                        max_nfev=10000)
    _params = res.x
    return _params.copy()

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_features(row) @ _params)
