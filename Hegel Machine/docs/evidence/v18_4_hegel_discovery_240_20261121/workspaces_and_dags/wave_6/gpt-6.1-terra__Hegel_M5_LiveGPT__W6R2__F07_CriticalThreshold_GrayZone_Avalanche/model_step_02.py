import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(p, row):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * np.sqrt(u) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    y = np.array [float(r['y']) for r in train_rows]
    def residual(p):
        return np.array([_value(p, r) for r in train_rows]) - y
    best = None
    for threshold in (2.50, 2.58, 2.67):
        result = least_squares(residual, [6.5, 0.7, 0.3, 2.0, threshold], bounds=([1.0, 0.05, -2.0, 0.0, 2.50], [15.0, 0.95, 2.0, 20.0, 2.68]), max_nfev=5000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_value(_params, row))