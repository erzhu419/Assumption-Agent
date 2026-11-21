import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _value(row, p):
    vmax, rho, c_loc = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    un = u  # characteristic tau scale is 1.0
    return vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un)) + c_loc * math.sqrt(un)

def fit(train_rows):
    global _P
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    best = None
    for start in ([4.0, 0.7, 0.2], [6.0, 0.5, 0.5], [3.0, 0.9, 0.0]):
        result = least_squares(residual, np.asarray(start), bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0]), max_nfev=10000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _P = np.asarray(best[1], dtype=float)
    return _P

def predict(row):
    if _P is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_value(row, _P))