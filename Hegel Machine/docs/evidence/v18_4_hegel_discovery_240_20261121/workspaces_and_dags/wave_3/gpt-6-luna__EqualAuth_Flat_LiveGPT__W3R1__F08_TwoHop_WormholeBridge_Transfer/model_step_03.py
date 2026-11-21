import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return u, s

def _model(p, row):
    u, s = _features(row)
    # Saturating hub response with a signed gradient-driven bridge flux.
    return p[0] * u / (p[1] + u) + p[2] * (1.0 - s) + p[3]

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    x0 = np.array([5.0, 1.0, 1.0, 0.0], dtype=float)
    result = least_squares(lambda p: np.asarray([_model(p, r) for r in train_rows]) - y,
                           x0, bounds=([0.0, 1e-6, -10.0, -10.0], [20.0, 20.0, 10.0, 10.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(_params, row))