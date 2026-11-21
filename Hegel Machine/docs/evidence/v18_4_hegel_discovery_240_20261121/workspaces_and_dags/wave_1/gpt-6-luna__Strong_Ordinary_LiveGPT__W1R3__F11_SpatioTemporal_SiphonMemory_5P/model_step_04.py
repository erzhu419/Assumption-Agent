import math
import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    vmax, rho, c_loc = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(max(u, 0.0))

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_pred(r, p) for r in train_rows]) - y
    starts = [np.array([2.0, 0.5, c]) for c in (-1.0, 0.0, 1.0)]
    fits = [least_squares(residual, p, max_nfev=20000) for p in starts]
    best = min(fits, key=lambda z: float(np.dot(z.fun, z.fun)))
    return best.x

def predict(row):
    return float(_pred(row, _FIT_PARAMS))

_FIT_PARAMS = None

def fit(train_rows):
    global _FIT_PARAMS
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_pred(r, p) for r in train_rows]) - y
    starts = [np.array([2.0, 0.5, c]) for c in (-1.0, 0.0, 1.0)]
    fits = [least_squares(residual, p, max_nfev=20000) for p in starts]
    _FIT_PARAMS = min(fits, key=lambda z: float(np.dot(z.fun, z.fun))).x
    return _FIT_PARAMS