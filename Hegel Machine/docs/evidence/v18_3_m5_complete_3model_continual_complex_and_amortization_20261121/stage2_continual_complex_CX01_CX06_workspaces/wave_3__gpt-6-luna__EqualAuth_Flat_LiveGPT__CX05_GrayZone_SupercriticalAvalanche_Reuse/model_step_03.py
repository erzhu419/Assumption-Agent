import math
import numpy as np
from scipy.optimize import least_squares

_FIT_PARAMS = None

def _model(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _FIT_PARAMS
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - ys
    starts = [
        [4.0, 0.5, 0.5, 1.0, 1.4],
        [3.0, 0.2, 1.0, 2.0, 1.3],
        [5.0, 0.8, 0.2, 0.5, 1.5]
    ]
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=([0.0, 0.0, -10.0, 0.0, 0.5], [20.0, 1.0, 10.0, 20.0, 3.0]), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _FIT_PARAMS = np.asarray(best[1], dtype=float)
    return _FIT_PARAMS

def predict(row):
    if _FIT_PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _FIT_PARAMS))