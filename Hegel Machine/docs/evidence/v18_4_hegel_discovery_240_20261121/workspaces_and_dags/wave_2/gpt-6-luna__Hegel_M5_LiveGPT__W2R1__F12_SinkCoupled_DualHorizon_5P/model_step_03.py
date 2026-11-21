import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, rho, amp_adapt, tau_fast, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (1.0 - math.exp(-H / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-H / tau_slow))

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    starts = [np.array([6.0, 0.70, 0.5, 0.5, 10.0]), np.array([5.5, 0.70, 0.4, 0.3, 8.0]), np.array([7.0, 0.65, 0.3, 0.8, 15.0])]
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, bounds=([0.01, 0.001, 0.001, 0.01, 0.1], [30.0, 0.999, 5.0, 5.0, 100.0]), max_nfev=5000)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    return best.x

def predict(row):
    return float(_value(row, fit(train_rows)))