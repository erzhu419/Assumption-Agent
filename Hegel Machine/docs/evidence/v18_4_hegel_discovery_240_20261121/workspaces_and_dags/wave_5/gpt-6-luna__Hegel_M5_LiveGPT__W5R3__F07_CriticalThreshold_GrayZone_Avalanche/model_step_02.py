import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(p, row):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])  # tau_scale = 1.0
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(max(u, 0.0)) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_predict_with(p, r) for r in rows]) - y
    p0 = np.array([6.5, 0.70, 0.1, 1.0, 2.59])
    lo = np.array([1.0, 0.05, -2.0, 0.0, 2.50])
    hi = np.array([15.0, 0.95, 2.0, 10.0, 2.68])
    result = least_squares(residual, p0, bounds=(lo, hi), max_nfev=20000)
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_predict_with(_params, row))