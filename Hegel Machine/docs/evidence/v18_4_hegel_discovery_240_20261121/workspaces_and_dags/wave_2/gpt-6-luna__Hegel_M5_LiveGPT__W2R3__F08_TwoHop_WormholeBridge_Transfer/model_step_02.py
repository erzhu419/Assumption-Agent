import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u = float(row['u_now'])
    g = float(row['s_grad'])
    return u, g

def _predict_params(row, p):
    u, g = _features(row)
    a, k, b, c = p
    return a * (1.0 - math.exp(-k * u)) + (b + c * u) * (1.0 - g)

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([2.5, 1.0, 0.0, 0.0]), max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_predict_params(row, _PARAMS))