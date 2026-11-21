import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_TAU = 0.02

def _value(row, p):
    j0, beta, rho, c_sqrt = p
    u = float(row['u_now']) / _TAU
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(u, 0.0))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    ys = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - ys
    best = None
    for start in ([1.0, 1.0, 0.5, 0.0], [2.0, 0.2, 0.2, 0.2], [1.0, 3.0, 0.7, -0.2]):
        res = least_squares(residual, np.asarray(start, dtype=float), max_nfev=20000)
        score = float(np.dot(res.fun, res.fun))
        if best is None or score < best[0]:
            best = (score, res.x)
    _PARAMS = best[1]
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _PARAMS))