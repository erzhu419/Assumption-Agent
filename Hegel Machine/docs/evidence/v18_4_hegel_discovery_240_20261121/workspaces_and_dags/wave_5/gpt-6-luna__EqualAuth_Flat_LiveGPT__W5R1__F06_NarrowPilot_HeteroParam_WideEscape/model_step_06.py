import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(row, p):
    j0, beta, rho, c_sqrt = p
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    un = u
    return j0 * un * math.exp(-beta / (0.5 + un)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(un)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_predict_with(r, p) for r in rows]) - y
    starts = [
        [5.0, 1.0, 0.5, 0.5],
        [10.0, 3.0, 0.8, 1.0],
        [3.0, 0.2, 0.2, 0.0],
        [8.0, 2.0, 1.2, 2.0]
    ]
    best = None
    for x0 in starts:
        result = least_squares(residual, x0=x0, bounds=([0.0, 0.0, -5.0, -10.0], [100.0, 30.0, 5.0, 10.0]), max_nfev=20000)
        score = float(np.mean(result.fun ** 2))
        if best is None or score < best[0]:
            best = (score, result.x)
    _params = np.asarray(best[1], dtype=float)
    return _params

def predict(row):
    if _params is None:
        raise ValueError('Call fit(train_rows) before predict(row).')
    return float(_predict_with(row, _params))