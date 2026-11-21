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
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_response(r, p) for r in train_rows]) - y
    starts = [
        [5.0, 0.7, 0.0, 2.0, 2.6],
        [4.0, 0.5, 0.2, 1.0, 2.5],
        [8.0, 0.8, -0.2, 3.0, 2.7],
    ]
    best = None
    for p0 in starts:
        result = least_squares(residual, p0,
            bounds=([0.0, 0.0, -10.0, 0.0, 2.0], [100.0, 1.0, 10.0, 20.0, 4.0]),
            max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _PARAMS = best[1]
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_response(row, _PARAMS))