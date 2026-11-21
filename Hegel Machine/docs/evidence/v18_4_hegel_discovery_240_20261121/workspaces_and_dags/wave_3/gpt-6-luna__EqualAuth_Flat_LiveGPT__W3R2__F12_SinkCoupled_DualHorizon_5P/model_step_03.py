import math
import numpy as np
from scipy.optimize import least_squares

def _model(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            * (1.0 - math.exp(-h / tau_fast))
            - amp_adapt * u * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    lower = np.array([0.0, 0.0, 1e-4, 0.0, 1e-4])
    upper = np.array([100.0, 1.0, 100.0, 100.0, 1000.0])
    starts = [
        [4.0, 0.4, 0.5, 0.3, 12.0],
        [3.0, 0.2, 1.0, 0.2, 20.0],
        [5.0, 0.6, 0.3, 0.1, 8.0],
    ]
    results = [least_squares(residual, np.asarray(p), bounds=(lower, upper), max_nfev=20000) for p in starts]
    return min(results, key=lambda result: float(np.dot(result.fun, result.fun))).x

def predict(row):
    return float(_model(row, fit(train_rows)))