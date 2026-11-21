import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _response(row, p):
    a, b, k, d = p
    u = max(0.0, float(row['u_now']))
    return a + b * (1.0 - math.exp(-k * u)) + d * float(row['s_grad'])

def fit(train_rows):
    global _PARAMS
    def residual(p):
        return np.asarray([_response(r, p) - float(r['y']) for r in train_rows])
    starts = [
        [0.0, 3.0, 0.7, -0.8],
        [0.5, 2.5, 1.0, -1.0],
        [-1.0, 4.0, 0.5, -0.7],
    ]
    fits = [least_squares(residual, x0, max_nfev=20000) for x0 in starts]
    best = min(fits, key=lambda result: float(np.dot(result.fun, result.fun)))
    _PARAMS = np.asarray(best.x, dtype=float)
    return _PARAMS.copy()

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_response(row, _PARAMS))