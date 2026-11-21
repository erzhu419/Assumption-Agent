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
    rows = list(train_rows)
    def residual(p):
        return np.array([_value(r, p) - float(r['y']) for r in rows])
    bounds = ([0.0, 0.0, -np.inf, 0.0, 2.50], [np.inf, 1.5, np.inf, np.inf, 2.68])
    fits = [least_squares(residual, [v, rho, c, k, threshold], bounds=bounds, max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12) for v, rho, c, k, threshold in [(5.0, 0.7, 1.0, 2.0, 2.55), (10.0, 0.7, 0.0, 3.0, 2.65), (3.0, 0.5, 2.0, 1.0, 2.60)]]
    _params = min(fits, key=lambda result: np.dot(result.fun, result.fun)).x

def predict(row):
    return float(_value(row, _params))