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
    bounds = ([1.0, 0.05, -2.0, 0.0, 1.3], [15.0, 0.95, 2.0, 20.0, 1.5])
    candidates = [least_squares(residual, [6.5, 0.7, 0.3, 2.0, threshold], bounds=bounds, max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12) for threshold in (1.31, 1.4, 1.49)]
    _params = min(candidates, key=lambda result: np.dot(result.fun, result.fun)).x

def predict(row):
    return float(_value(row, _params))