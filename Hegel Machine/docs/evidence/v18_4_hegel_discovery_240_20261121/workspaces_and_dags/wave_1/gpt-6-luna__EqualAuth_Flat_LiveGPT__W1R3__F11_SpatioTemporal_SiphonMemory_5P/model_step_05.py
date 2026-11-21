import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    vmax, rho, c_loc = p
    u = max(1e-4, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u)

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_model(r, p) for r in train_rows]) - y
    starts = ([2.0, 0.5, 0.2], [1.0, 0.8, 0.5], [3.0, 0.2, -0.2])
    fits = [least_squares(residual, np.array(p, dtype=float), max_nfev=10000) for p in starts]
    _PARAMS = min(fits, key=lambda q: float(np.sum(q.fun ** 2))).x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _PARAMS))