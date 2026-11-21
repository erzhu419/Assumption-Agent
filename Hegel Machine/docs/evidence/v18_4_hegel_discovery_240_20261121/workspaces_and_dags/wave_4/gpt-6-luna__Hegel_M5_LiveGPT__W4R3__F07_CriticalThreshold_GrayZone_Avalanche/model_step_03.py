import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(max(u, 0.0)) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    starts = [np.array([7.0, 0.70, 0.2, 2.0, 2.6]), np.array([6.0, 0.72, 0.5, 1.5, 2.5]), np.array([8.0, 0.68, -0.2, 2.5, 2.7])]
    fits = [least_squares(residual, x0, bounds=([0.01, 0.0, -5.0, 0.0, 2.0], [30.0, 0.999, 5.0, 20.0, 3.5]), max_nfev=20000) for x0 in starts]
    _params = min(fits, key=lambda z: float(np.sum(z.fun ** 2))).x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))