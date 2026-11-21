import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _response(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(max(u, 0.0)) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_response(r, p) for r in rows]) - y
    starts = [
        [5.0, 0.7, 0.0, 1.0, 2.6],
        [4.0, 0.5, 0.3, 0.8, 2.5],
        [6.0, 0.8, -0.3, 1.5, 2.7],
    ]
    fits = [least_squares(residual, x0, bounds=([0.0, 0.0, -10.0, 0.0, 2.0], [100.0, 1.5, 10.0, 20.0, 3.5]), max_nfev=20000) for x0 in starts]
    best = min(fits, key=lambda result: float(np.dot(result.fun, result.fun)))
    _PARAMS = best.x.copy()
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_response(row, _PARAMS))