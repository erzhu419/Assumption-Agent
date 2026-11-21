import math
import numpy as np
from scipy.optimize import least_squares

_tau = 0.02
_params = None

def _features(row):
    u = float(row['u_now']) / _tau
    s = float(row['s_grad'])
    return u, s

def _model(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u, s = _features(row)
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_model(r, p) for r in rows]) - y
    starts = [
        [5.0, 1.0, 0.70, 0.05],
        [8.0, 2.0, 0.70, 0.10],
        [3.0, 0.5, 0.70, 0.02],
    ]
    best = None
    for start in starts:
        result = least_squares(residual, start,
            bounds=([0.0, 0.0, 0.05, 0.0], [100.0, 100.0, 0.95, 100.0]),
            max_nfev=20000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _params = best.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_model(row, _params))