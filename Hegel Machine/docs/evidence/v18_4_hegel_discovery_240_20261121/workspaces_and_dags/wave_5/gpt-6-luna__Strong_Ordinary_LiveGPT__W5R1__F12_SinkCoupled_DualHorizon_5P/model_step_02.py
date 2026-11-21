import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, rho, amp_adapt, tau_fast, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    sink = 1.0 - rho * s / (1.0 + s)
    base = vmax * sink * (u / (1.2 + u)) * (1.0 - math.exp(-H / tau_fast))
    adapt = amp_adapt * u * (1.0 - math.exp(-H / tau_slow))
    return base - adapt

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    starts = [
        [3.0, 0.7, 0.4, 0.5, 10.0],
        [2.0, 0.3, 0.3, 0.2, 20.0],
        [4.0, 0.9, 0.6, 1.0, 30.0],
    ]
    lo = [0.0, 0.0, 0.0, 1e-4, 0.01]
    hi = [100.0, 1.0, 100.0, 100.0, 1000.0]
    fits = [least_squares(residual, x0, bounds=(lo, hi), max_nfev=20000) for x0 in starts]
    best = min(fits, key=lambda result: float(np.dot(result.fun, result.fun)))
    return {'params': [float(v) for v in best.x]}

def predict(row):
    return float(_value(row, fit_params))