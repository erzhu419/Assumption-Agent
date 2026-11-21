import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(u) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    best = None
    starts = [
        [5.0, 0.7, 0.0, 1.0, 2.6],
        [3.0, 0.5, 0.2, 0.5, 2.5],
        [8.0, 0.8, -0.5, 2.0, 2.7],
    ]
    for x0 in starts:
        result = least_squares(residual, x0=x0,
            bounds=([0.0, 0.0, -10.0, 0.0, 2.0], [100.0, 1.0, 10.0, 20.0, 3.2]),
            max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _PARAMS = best[1].copy()
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_value(row, _PARAMS))