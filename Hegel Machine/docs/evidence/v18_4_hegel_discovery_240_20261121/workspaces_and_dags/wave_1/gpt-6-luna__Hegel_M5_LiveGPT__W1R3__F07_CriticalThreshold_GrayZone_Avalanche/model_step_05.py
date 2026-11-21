import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = max(1e-4, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * (u*u / (k_half + u*u)) / (1.0 + alpha_inh*s) - c_leak*u

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - y
    best = None
    for start in ([3.0, 1.0, 0.5, 0.5], [6.0, 4.0, 1.0, 0.8], [10.0, 10.0, 2.0, 1.0]):
        res = least_squares(residual, start, bounds=([1e-8, 1e-8, 1e-8, 0.0], [100.0, 100.0, 100.0, 100.0]), max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    return float(_value(row, fit(_TRAIN_ROWS)))