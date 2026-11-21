import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    vmax, km, c_sqrt, beta = p
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * u / (km + u) + c_sqrt * math.sqrt(u) + beta * (s - 1.0)

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_model(r, p) for r in train_rows]) - y
    starts = ([6.0, 1.0, 0.5, 0.0], [4.0, 0.5, 1.0, -0.1], [8.0, 2.0, 0.0, 0.1])
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=([0.0, 1e-6, -10.0, -10.0], [100.0, 100.0, 10.0, 10.0]))
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _PARAMS = best.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_model(row, _PARAMS))