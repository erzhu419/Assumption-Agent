import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_params(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])  # tau_scale = 1.0
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(max(0.0, u)) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in rows]) - y
    p0 = np.array([5.0, 0.7, 0.2, 2.0, 2.6])
    lower = np.array([0.0, 0.0, -10.0, 0.0, 2.0])
    upper = np.array([30.0, 1.0, 10.0, 10.0, 3.2])
    result = least_squares(residual, p0, bounds=(lower, upper), max_nfev=50000)
    _PARAMS = result.x
    return _PARAMS.copy()

def predict(row):
    if _PARAMS is None:
        raise ValueError('Call fit(train_rows) before predict(row)')
    return float(_predict_params(row, _PARAMS))