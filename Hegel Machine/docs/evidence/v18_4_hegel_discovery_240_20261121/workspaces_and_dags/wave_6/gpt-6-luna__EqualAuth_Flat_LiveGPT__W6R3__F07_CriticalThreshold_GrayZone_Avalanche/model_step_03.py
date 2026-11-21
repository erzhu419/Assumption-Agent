import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(max(u, 0.0)) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    starts = [
        [5.0, 0.8, 0.3, 2.0, 2.6],
        [4.0, 0.7, 0.5, 1.5, 2.5],
        [6.0, 0.9, 0.1, 3.0, 2.7],
    ]
    lower = [0.0, 0.0, -10.0, 0.0, 2.0]
    upper = [50.0, 5.0, 10.0, 20.0, 3.2]
    sols = [least_squares(residual, x0, bounds=(lower, upper), max_nfev=20000) for x0 in starts]
    _params = min(sols, key=lambda z: float(np.dot(z.fun, z.fun))).x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _params))