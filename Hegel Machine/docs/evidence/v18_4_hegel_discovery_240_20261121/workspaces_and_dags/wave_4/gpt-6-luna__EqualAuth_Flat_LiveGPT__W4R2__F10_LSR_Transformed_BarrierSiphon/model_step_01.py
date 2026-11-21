import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u = max(float(row['u_now']), 0.0)
    s = float(row['s_grad'])
    return u, s

def _model(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u, s = _features(row)
    inhibition = 1.0 - rho * s / (1.0 + s)
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * inhibition + c_sqrt * math.sqrt(u)

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in rows]) - y
    starts = [np.array([1.0, 1.0, 0.5, 0.5]), np.array([2.0, 0.5, 0.2, 0.2]), np.array([1.0, 2.0, 0.8, 1.0])]
    fits = [least_squares(residual, x0, bounds=([0.0, 0.0, 0.0, 0.0], [100.0, 100.0, 10.0, 100.0])) for x0 in starts]
    best = min(fits, key=lambda result: float(np.dot(result.fun, result.fun)))
    _PARAMS = best.x
    return {'j0': float(_PARAMS[0]), 'beta_bar': float(_PARAMS[1]), 'rho': float(_PARAMS[2]), 'c_sqrt': float(_PARAMS[3])}

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_model(row, _PARAMS))