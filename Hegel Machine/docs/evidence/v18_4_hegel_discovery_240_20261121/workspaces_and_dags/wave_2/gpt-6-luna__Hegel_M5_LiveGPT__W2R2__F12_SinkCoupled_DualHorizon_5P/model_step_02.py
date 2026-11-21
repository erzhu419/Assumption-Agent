import math
import numpy as np
from scipy.optimize import least_squares

def _model(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    s = float(row['s_grad'])
    u = float(row['u_now'])
    H = float(row['H'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (1.0 - math.exp(-H / tau_fast))
            - amp_adapt * u * (1.0 - math.exp(-H / tau_slow)))

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    starts = [
        [6.5, 0.70, 1.0, 0.8, 12.0],
        [5.0, 0.70, 0.5, 0.5, 8.0],
        [8.0, 0.65, 2.0, 0.4, 18.0],
    ]
    fits = [least_squares(residual, x0, bounds=([0.01, 0.0, 0.01, 0.0, 0.02], [30.0, 0.999, 20.0, 10.0, 200.0]), max_nfev=30000) for x0 in starts]
    return min(fits, key=lambda result: float(np.dot(result.fun, result.fun))).x

def predict(row):
    return float(_model(row, fit(train_rows)))