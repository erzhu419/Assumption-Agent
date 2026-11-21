import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = max(float(row['u_now']), 0.0)
    s = float(row['s_grad'])
    flux = j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s))
    return flux + c_sqrt * math.sqrt(u)

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    result = least_squares(residual, x0=[10.0, 0.5, 0.7, 0.0],
                           bounds=([0.0, 0.0, 0.0, -np.inf],
                                   [np.inf, 10.0, 1.0, np.inf]),
                           max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _PARAMS))