import numpy as np
from scipy.optimize import least_squares
import math

def fit(train_rows):
    rows = list(train_rows)
    def pred(p, r):
        vmax, rho, c_loc, k_aval, s_crit = p
        u = float(r['u_now'])
        s = float(r['s_grad'])
        base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
        return base + c_loc * math.sqrt(max(u, 0.0)) - k_aval * max(0.0, s - s_crit) * u
    def residual(p):
        return np.array([pred(p, r) - float(r['y']) for r in rows])
    p0 = np.array([6.0, 0.70, 0.0, 2.0, 1.3])
    lo = np.array([1.0, 0.05, -2.0, 0.0, 0.0])
    hi = np.array([15.0, 0.95, 2.0, 20.0, 3.0])
    result = least_squares(residual, p0, bounds=(lo, hi), max_nfev=30000)
    return result.x

def predict(row):
    vmax, rho, c_loc, k_aval, s_crit = fit_params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return float(base + c_loc * math.sqrt(max(u, 0.0)) - k_aval * max(0.0, s - s_crit) * u)
