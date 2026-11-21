import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u = float(row['u_now'])
    sg = float(row['s_grad'])
    x = float(row['x_now'])
    m = float(row['mem_conv'])
    h = float(row['H'])
    up = float(row['u_past'])
    sink_pool = (u / (1.2 + u)) * (1.0 - sg / (1.0 + sg))
    return np.array([sink_pool, math.sqrt(max(u, 0.0)), x, m, u - up, h * (x - m), 1.0], dtype=float)

def fit(train_rows):
    global _PARAMS
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    # Numerically fit coefficients; residual scaling keeps optimization well-conditioned.
    scale = np.maximum(np.linalg.norm(X, axis=0), 1e-8)
    res = least_squares(lambda q: (X / scale) @ q - y, np.zeros(X.shape[1]), max_nfev=10000)
    _PARAMS = res.x / scale
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_features(row) @ _PARAMS)
