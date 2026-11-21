import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            + c_loc * math.sqrt(max(0.0, u))
            - k_aval * max(0.0, s - s_crit) * u)

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_model(r, p) for r in rows]) - y
    starts = [
        [4.0, 0.5, 0.2, 1.0, 2.6],
        [3.0, 0.3, 0.5, 0.8, 2.5],
        [6.0, 0.7, 0.1, 1.5, 2.7],
    ]
    best = None
    for x0 in starts:
        res = least_squares(residual, x0, bounds=([0, 0, 0, 0, 0], [100, 5, 20, 20, 10]), max_nfev=20000)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    _PARAMS = best.x
    return _PARAMS.copy()

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_model(row, _PARAMS))