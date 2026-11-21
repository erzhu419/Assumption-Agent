import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now']) / 3.0
    s = float(row['s_grad'])
    return u, s

def _model(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u, s = _features(row)
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(u, 0.0))

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    starts = [
        [1.0, 1.0, 0.4, 0.1],
        [2.0, 3.0, 0.8, 0.5],
        [0.5, 0.1, 0.1, 1.0],
    ]
    fits = [least_squares(residual, x0, bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, 20.0, 1.0, np.inf]), max_nfev=20000) for x0 in starts]
    best = min(fits, key=lambda z: float(np.dot(z.fun, z.fun)))
    _params = best.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _params))