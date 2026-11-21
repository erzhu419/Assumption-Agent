import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * math.sqrt(u) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_value(row, p) - float(row['y']) for row in train_rows])
    best = None
    for threshold in (2.50, 2.59, 2.68):
        result = least_squares(residual, [5.0, 0.7, 0.5, 2.0, threshold], bounds=([0.0, 0.0, -10.0, 0.0, 2.44], [100.0, 2.0, 10.0, 100.0, 2.78]), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _params = best.x

def predict(row):
    return float(_value(row, _params))
