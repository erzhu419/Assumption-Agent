import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, rho, c_loc = p
    u = max(1e-9, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u)

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - y
    best = None
    for start in ([3.0, 0.7, 0.0], [5.0, 0.3, 0.5], [4.0, 0.9, -0.5], [2.0, 0.5, 1.0]):
        result = least_squares(residual, np.array(start, dtype=float), bounds=([0.0, 0.0, -10.0], [50.0, 1.0, 10.0]), max_nfev=20000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    return best.x

def predict(row):
    return float(_value(row, _params))

_params = None
_fit_impl = fit
def fit(train_rows):
    global _params
    _params = _fit_impl(train_rows)
    return _params
