import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_with(p, row):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    local = c_loc * math.sqrt(u)
    avalanche = k_aval * max(0.0, s - s_crit) * u
    return base + local - avalanche

def fit(train_rows):
    global _PARAMS
    ys = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_with(p, r) for r in train_rows]) - ys
    starts = [
        [5.0, 0.5, 0.5, 1.0, 2.6],
        [4.0, 0.3, 0.8, 1.5, 2.5],
        [6.0, 0.7, 0.2, 0.8, 2.7],
    ]
    fits = [least_squares(residual, x0, bounds=([0.0, 0.0, 0.0, 0.0, 2.0], [100.0, 5.0, 20.0, 20.0, 4.0]), max_nfev=20000) for x0 in starts]
    best = min(fits, key=lambda z: float(np.sum(z.fun ** 2)))
    _PARAMS = best.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_with(_PARAMS, row))