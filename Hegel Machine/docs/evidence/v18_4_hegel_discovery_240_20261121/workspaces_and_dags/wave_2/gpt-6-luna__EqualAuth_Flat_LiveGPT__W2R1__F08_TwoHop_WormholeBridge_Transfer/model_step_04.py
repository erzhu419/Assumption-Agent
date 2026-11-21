import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, row):
    A, B, k, c = p
    u = max(float(row['u_now']), 0.0)
    g = float(row['s_grad'])
    return c + A * u / (B + u) * math.exp(-k * (g - 1.0))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_model(p, r) for r in rows]) - y
    result = least_squares(residual, x0=np.array([3.0, 1.0, 0.4, 0.0]), bounds=([0.0, 1e-6, -3.0, -5.0], [20.0, 20.0, 3.0, 5.0]), max_nfev=20000)
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(_params, row))