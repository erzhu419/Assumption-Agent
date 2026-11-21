import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    starts = [
        [5.0, 0.5, 0.5, 1.0, 2.6],
        [4.0, 0.3, 0.8, 1.5, 2.5],
        [6.0, 0.7, 0.2, 0.8, 2.7],
    ]
    fits = [least_squares(residual, x0, bounds=([0.0, 0.0, 0.0, 0.0, 0.0], [100.0, 5.0, 20.0, 20.0, 10.0]), max_nfev=20000) for x0 in starts]
    best = min(fits, key=lambda z: float(np.dot(z.fun, z.fun)))
    _params = best.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_model(row, _params))