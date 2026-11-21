import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * np.sqrt(u) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - y
    best = None
    for threshold in (2.48, 2.58, 2.68, 2.75):
        result = least_squares(residual, [5.0, 0.7, 1.0, 1.5, threshold], bounds=([0.0, 0.0, -20.0, 0.0, 2.44], [100.0, 2.0, 20.0, 20.0, 2.78]), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_value(row, _params))