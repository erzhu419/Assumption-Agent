import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
            + c_loc * math.sqrt(max(0.0, u))
            - k_aval * max(0.0, s - s_crit) * u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_value(r, p) - float(r['y']) for r in rows])
    bounds = ([1.0, 0.05, -2.0, 0.0, 0.0],
              [15.0, 0.95, 2.0, 20.0, 5.0])
    fits = [least_squares(residual, [6.5, 0.7, 0.1, k, threshold],
                          bounds=bounds, max_nfev=10000,
                          ftol=1e-12, xtol=1e-12, gtol=1e-12)
            for threshold in (2.35, 2.5, 2.7, 3.0)
            for k in (1.0, 2.0)]
    _params = min(fits, key=lambda result: np.dot(result.fun, result.fun)).x

def predict(row):
    return float(_value(row, _params))