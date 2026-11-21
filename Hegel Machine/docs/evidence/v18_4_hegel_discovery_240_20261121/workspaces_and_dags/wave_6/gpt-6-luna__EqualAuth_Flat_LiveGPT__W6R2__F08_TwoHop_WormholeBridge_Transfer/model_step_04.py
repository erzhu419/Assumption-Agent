import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p, sign):
    vmax, rho, c_loc = p
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + sign * c_loc * math.sqrt(u)

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    best = None
    for sign in (1.0, -1.0):
        def residual(p):
            return np.array([_value(r, p, sign) - float(r['y']) for r in rows])
        result = least_squares(residual, x0=np.array([4.0, 0.7, 0.2]), bounds=([0.0, 0.0, 0.0], [100.0, 0.999, 100.0]))
        score = float(np.mean(residual(result.x) ** 2))
        if best is None or score < best[0]:
            best = (score, sign, result.x)
    _PARAMS = (best[1], np.asarray(best[2], dtype=float))
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    sign, params = _PARAMS
    return float(_value(row, params, sign))