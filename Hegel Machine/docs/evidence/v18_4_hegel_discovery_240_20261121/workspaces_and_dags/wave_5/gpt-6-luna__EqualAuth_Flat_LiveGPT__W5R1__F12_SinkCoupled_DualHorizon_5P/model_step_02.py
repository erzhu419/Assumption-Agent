import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, params):
    vmax, rho, tau_fast, amp_adapt, tau_slow = params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * (1.0 - math.exp(-H / tau_fast))
            - amp_adapt * u * (1.0 - math.exp(-H / tau_slow)))

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in rows], dtype=float) - y
    best = None
    starts = [
        [4.0, 0.4, 0.5, 0.2, 10.0],
        [3.0, 0.2, 0.3, 0.15, 20.0],
        [6.0, 0.6, 1.0, 0.1, 8.0],
        [2.0, 0.1, 0.2, 0.3, 30.0],
    ]
    lower = [0.0, 0.0, 1e-4, 0.0, 1e-3]
    upper = [100.0, 1.0, 100.0, 10.0, 500.0]
    for x0 in starts:
        result = least_squares(residual, x0, bounds=(lower, upper), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    return np.asarray(best[1], dtype=float)

def predict(row):
    return float(_value(row, fit_params))