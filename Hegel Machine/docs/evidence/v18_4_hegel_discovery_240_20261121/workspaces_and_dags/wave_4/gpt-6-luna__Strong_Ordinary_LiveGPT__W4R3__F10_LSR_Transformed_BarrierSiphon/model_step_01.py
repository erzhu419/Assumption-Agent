import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_TAU_SCALE = 0.02

def _value(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u_n = max(float(row['u_now']) / _TAU_SCALE, 0.0)
    s = float(row['s_grad'])
    return j0 * u_n * math.exp(-beta_bar / (0.5 + u_n)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u_n)

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - y
    starts = [
        [1.0, 1.0, 0.5, 0.0],
        [2.0, 0.2, 0.7, 0.1],
        [5.0, 2.0, 0.3, -0.1],
        [0.5, 0.05, 0.8, 0.5],
    ]
    best = None
    for start in starts:
        res = least_squares(residual, start, bounds=([0.0, 0.0, -5.0, -20.0], [1000.0, 100.0, 5.0, 20.0]), max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    _PARAMS = best.x
    return _PARAMS.copy()

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _PARAMS))
