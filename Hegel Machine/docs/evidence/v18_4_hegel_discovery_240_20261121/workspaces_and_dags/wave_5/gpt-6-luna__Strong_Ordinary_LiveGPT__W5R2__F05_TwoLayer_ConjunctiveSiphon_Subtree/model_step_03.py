import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c_loc = p
    u = max(float(row['u_now']), 0.0)
    s = max(float(row['s_grad']), 0.0)
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(u)

def fit(train_rows):
    global _params
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - y
    starts = ([4.0, 0.7, 0.2], [4.0, 0.7, -0.2], [2.0, 0.3, 0.0])
    results = [least_squares(residual, x0, bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0])) for x0 in starts]
    best = min(results, key=lambda result: float(np.sum(result.fun ** 2)))
    _params = best.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))