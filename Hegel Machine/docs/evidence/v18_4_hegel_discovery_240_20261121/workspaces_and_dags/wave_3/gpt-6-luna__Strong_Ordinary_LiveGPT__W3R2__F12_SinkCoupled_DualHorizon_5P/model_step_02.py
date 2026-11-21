import math
import numpy as np
from scipy.optimize import least_squares

def _model(p, row):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base * (1.0 - math.exp(-H / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-H / tau_slow))

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_model(p, r) for r in train_rows]) - y
    x0 = np.array([3.0, 0.7, 1.0, 0.5, 12.0], dtype=float)
    lo = np.array([1e-8, 0.0, 1e-4, 0.0, 1e-3])
    hi = np.array([100.0, 1.0, 100.0, 100.0, 500.0])
    result = least_squares(residual, x0, bounds=(lo, hi), max_nfev=20000)
    return result.x

def predict(row):
    p = fit(train_rows)
    return float(_model(p, row))