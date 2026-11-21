import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])  # tau_scale = 1.0
    s = float(row['s_grad'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            + c_loc * math.sqrt(max(u, 0.0))
            - k_aval * max(0.0, s - s_crit) * u)

def fit(train_rows):
    global _params
    lower = np.array([1.0, 0.05, -2.0, 0.0, 2.50])
    upper = np.array([15.0, 0.95, 2.0, 10.0, 2.68])
    x0 = np.array([6.5, 0.70, 0.2, 1.0, 2.60])
    def residual(p):
        return np.array([_value(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0, bounds=(lower, upper), max_nfev=20000)
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_value(row, _params))