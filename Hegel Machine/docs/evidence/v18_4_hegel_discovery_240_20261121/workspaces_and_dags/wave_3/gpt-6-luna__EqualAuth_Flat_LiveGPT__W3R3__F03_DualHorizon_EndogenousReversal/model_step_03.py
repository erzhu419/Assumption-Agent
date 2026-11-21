import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    H = float(row['H'])
    s = float(row.get('s_grad') or 0.0)
    sink = 1.0 - p[4] * s / (1.0 + s) if len(p) > 4 else 1.0
    return vmax * sink * (u / (1.2 + u)) * (1.0 - math.exp(-H / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-H / tau_slow))

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - y
    best = None
    starts = [[3.0, .2, 1.0, 10.0], [5.0, .5, 1.0, 20.0], [2.0, .1, .5, 8.0]]
    for p0 in starts:
        result = least_squares(residual, p0, bounds=([0.0, 1e-4, 0.0, .01], [100.0, 10.0, 100.0, 500.0]), max_nfev=30000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    return best.x.tolist()

def predict(row):
    p = fit([row])
    return float(_value(row, p))