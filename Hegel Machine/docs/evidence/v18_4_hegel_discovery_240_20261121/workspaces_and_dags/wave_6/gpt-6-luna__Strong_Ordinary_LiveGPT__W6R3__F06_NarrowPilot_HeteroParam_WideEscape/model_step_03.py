import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u)

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in rows]) - y
    starts = [
        [3.0, 0.5, 0.5, 0.5],
        [8.0, 1.0, 0.7, 0.2],
        [2.0, 0.1, 0.3, 1.0],
    ]
    fits = [least_squares(residual, x0, bounds=([0.0, 0.0, 0.0, -10.0], [100.0, 20.0, 1.0, 10.0]), max_nfev=20000) for x0 in starts]
    best = min(fits, key=lambda result: float(np.dot(result.fun, result.fun)))
    _PARAMS = best.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_model(row, _PARAMS))
