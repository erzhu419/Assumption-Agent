import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    base = j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s))
    return base + c_sqrt * math.sqrt(u)

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - y
    best = None
    for start in ([3.0, 1.0, 0.7, -1.0], [5.0, 2.0, 0.5, -2.0], [1.0, 0.2, 1.0, -0.5]):
        res = least_squares(residual, np.array(start), bounds=([-100.0, -20.0, -20.0, -100.0], [100.0, 20.0, 20.0, 100.0]), max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    return float(_value(row, _PARAMS))

_PARAMS = np.array([1.0, 1.0, 0.7, -1.0])
# The harness fits first; retain the fitted parameters for subsequent predict calls.
_ORIGINAL_FIT = fit
def fit(train_rows):
    global _PARAMS
    _PARAMS = _ORIGINAL_FIT(train_rows)
    return _PARAMS