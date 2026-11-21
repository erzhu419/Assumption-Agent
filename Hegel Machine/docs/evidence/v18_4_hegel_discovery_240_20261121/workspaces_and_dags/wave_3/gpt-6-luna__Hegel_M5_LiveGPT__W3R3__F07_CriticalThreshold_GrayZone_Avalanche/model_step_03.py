import math
import numpy as np
from scipy.optimize import least_squares

fit_params = None

def _raw(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    branch = c_loc * math.sqrt(max(0.0, u))
    avalanche = k_aval * max(0.0, s - s_crit) * u
    return base + branch - avalanche

def fit(train_rows):
    global fit_params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_raw(r, p) for r in rows]) - y
    p0 = np.array([6.5, 0.70, 0.1, 1.0, 2.59])
    lower = np.array([1.0, 0.05, -2.0, 0.0, 2.36])
    upper = np.array([15.0, 0.95, 2.0, 10.0, 3.5])
    result = least_squares(residual, p0, bounds=(lower, upper), max_nfev=20000)
    fit_params = result.x
    return fit_params

def predict(row) -> float:
    if fit_params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_raw(row, fit_params))