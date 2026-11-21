import math
import numpy as np
from scipy.optimize import least_squares

def _pred(p, row):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    x0 = np.array([6.0, 0.7, 0.2, 2.0, 2.55])
    lo = np.array([1.0, 0.05, -2.0, 0.0, 2.36])
    hi = np.array([15.0, 0.95, 2.0, 10.0, 3.2])
    res = least_squares(lambda p: np.array([_pred(p, r) for r in train_rows]) - y, x0, bounds=(lo, hi), max_nfev=20000)
    return res.x.tolist()

def predict(row):
    params = fit([row])
    return float(_pred(params, row))