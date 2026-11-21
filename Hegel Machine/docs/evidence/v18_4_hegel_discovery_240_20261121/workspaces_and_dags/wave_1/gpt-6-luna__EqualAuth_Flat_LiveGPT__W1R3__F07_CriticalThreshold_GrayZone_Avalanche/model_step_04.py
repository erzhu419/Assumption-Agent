import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    A, k, rho, C, q, t = p
    base = A * u / (k + u) * (1.0 - rho * s / (1.0 + s))
    z = max(-60.0, min(60.0, q * (s - t)))
    return base - C / (1.0 + math.exp(-z))

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_model(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=[5.0, 1.0, 0.5, 5.0, 3.0, 3.0],
        bounds=([0.0, 1e-4, -5.0, 0.0, 0.01, -10.0],
                [100.0, 100.0, 5.0, 100.0, 50.0, 20.0]), max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _PARAMS))