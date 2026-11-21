import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _pred(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(max(u, 0.0)) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_pred(r, p) for r in train_rows]) - y
    best = None
    for start in ([4.0, 0.5, 0.2, 1.0, 2.6], [5.0, 0.7, 0.0, 1.5, 2.55], [3.0, 0.3, 0.5, 0.8, 2.65]):
        result = least_squares(residual, np.asarray(start, dtype=float), bounds=([0.0, 0.0, -5.0, 0.0, 2.3], [20.0, 2.0, 5.0, 10.0, 2.9]), max_nfev=20000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _PARAMS = best.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_pred(row, _PARAMS))