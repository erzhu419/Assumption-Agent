import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    un = row['u_now'] / 0.005
    s = row['s_grad']
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))
    return base + c_loc * math.sqrt(max(un, 0.0)) - k_aval * max(0.0, s - s_crit) * un

def fit(train_rows):
    global _PARAMS
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=[1.0, 0.5, 0.01, 0.01, 1.4],
                           bounds=([0.0, 0.0, -1.0, 0.0, 1.3],
                                   [10.0, 1.0, 1.0, 1.0, 1.5]),
                           max_nfev=3000)
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _PARAMS))