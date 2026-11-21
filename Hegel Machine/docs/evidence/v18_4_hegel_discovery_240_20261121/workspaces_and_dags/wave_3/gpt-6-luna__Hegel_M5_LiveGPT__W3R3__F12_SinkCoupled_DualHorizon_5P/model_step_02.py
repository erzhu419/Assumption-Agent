import math
import numpy as np
from scipy.optimize import least_squares

def _model(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (1.0 - math.exp(-H / tau_fast))
    adapt = amp_adapt * u * (1.0 - math.exp(-H / tau_slow))
    return base - adapt

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_model(r, p) for r in rows]) - y
    lower = np.array([0.0, 0.0, 1e-4, 0.0, 1e-4])
    upper = np.array([30.0, 1.0, 100.0, 10.0, 500.0])
    starts = [
        [6.0, 0.7, 0.5, 0.2, 10.0],
        [5.0, 0.5, 1.0, 0.5, 20.0],
        [8.0, 0.8, 0.2, 0.1, 50.0],
        [3.0, 0.3, 2.0, 1.0, 100.0],
    ]
    best = None
    for start in starts:
        result = least_squares(residual, np.clip(start, lower + 1e-8, upper - 1e-8), bounds=(lower, upper), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    params = best[1]
    def predict(row):
        return float(_model(row, params))
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) first and use the returned predictor.')