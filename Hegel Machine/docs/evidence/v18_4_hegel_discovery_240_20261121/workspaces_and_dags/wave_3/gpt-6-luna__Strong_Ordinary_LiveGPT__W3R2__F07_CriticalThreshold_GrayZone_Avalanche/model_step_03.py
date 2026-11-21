import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    un = u  # characteristic tau scale is 1.0
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))
    return base + c_loc * math.sqrt(max(un, 0.0)) - k_aval * max(0.0, s - s_crit) * un

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_model(r, p) for r in train_rows]) - y
    starts = [
        [5.0, 0.7, 0.0, 2.0, 2.6],
        [4.0, 0.5, 0.2, 1.0, 2.5],
        [6.0, 0.8, -0.2, 3.0, 2.7],
    ]
    fits = [least_squares(residual, x0, bounds=([0.0, 0.0, -10.0, 0.0, 2.0], [100.0, 1.0, 10.0, 100.0, 3.5]), max_nfev=20000) for x0 in starts]
    best = min(fits, key=lambda z: float(np.dot(z.fun, z.fun)))
    _PARAMS = best.x
    return best.x

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _PARAMS))