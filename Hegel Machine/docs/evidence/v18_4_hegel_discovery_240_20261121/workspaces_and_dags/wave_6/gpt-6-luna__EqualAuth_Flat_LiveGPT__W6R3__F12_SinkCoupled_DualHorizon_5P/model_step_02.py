import math
import numpy as np
from scipy.optimize import least_squares

def _model(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    fast = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (1.0 - math.exp(-h / tau_fast))
    slow = amp_adapt * u * (1.0 - math.exp(-h / tau_slow))
    return fast - slow

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_model(r, p) for r in rows]) - y
    starts = [
        [4.0, 0.5, 0.5, 0.2, 15.0],
        [3.0, 0.2, 1.0, 0.15, 25.0],
        [5.0, 0.8, 0.3, 0.25, 10.0],
    ]
    best = None
    for start in starts:
        result = least_squares(residual, x0=start,
            bounds=([0.0, 0.0, 1e-5, 0.0, 1e-5], [20.0, 1.0, 100.0, 5.0, 500.0]),
            max_nfev=20000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    return np.asarray(best.x, dtype=float)

def predict(row):
    p = fit([row]) if not hasattr(row, '_fit_params') else row._fit_params
    return float(_model(row, p))